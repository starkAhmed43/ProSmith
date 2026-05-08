"""
feature_utils.py — ProSmith embedding helpers with persistent disk cache.

Protein encoder : ESM1b (esm1b_t33_650M_UR50S) → per-token [L, 1280] float32
SMILES encoder  : ChemBERTa-77M-MTR            → per-token logits [L, 600] float32

Cache layout (same strategy as CataPro emulator_bench):
  <cache_dir>/esm1b/<sha256>.npy    — shape (L, 1280), variable L ≤ 1018
  <cache_dir>/chembert/<sha256>.npy — shape (L, 600),  variable L ≤ 500

Sequences are truncated to 1018 tokens (ESM1b limit). ChemBERTa embeddings are
generated with tokenizer max_length=500 to mirror the original preprocessing,
then cropped to the training-time 256-token cap by the dataset loader.
"""

import gc
import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
PROT_MODEL_ID = "esm1b_t33_650M_UR50S"  # loaded via torch.hub / fair-esm
MOL_MODEL_ID = "DeepChem/ChemBERTa-77M-MTR"
CACHE_VERSION = "v2"

MAX_PROT_TOK = 1018   # ESM1b trained limit (truncate longer sequences)
MAX_SMILES_TOK = 500  # original smiles_embeddings.py tokenizer max_length
TRAIN_MAX_SMILES_TOK = 256  # original training-time cap in datautils.py


# ---------------------------------------------------------------------------
# Generic cache helpers
# ---------------------------------------------------------------------------

def _ensure_cache_root(cache_dir):
    if cache_dir is None:
        return None
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_key(namespace, value):
    text = f"{CACHE_VERSION}|{namespace}|{value}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_file(cache_root, namespace, key):
    subdir = cache_root / namespace
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{key}.npy"


def get_protein_cache_key(sequence):
    return _cache_key(PROT_MODEL_ID, str(sequence)[:MAX_PROT_TOK])


def get_smiles_cache_key(smiles):
    return _cache_key(MOL_MODEL_ID, str(smiles))


def get_protein_cache_path(cache_dir, sequence):
    cache_root = _ensure_cache_root(cache_dir)
    return _cache_file(cache_root, "esm1b", get_protein_cache_key(sequence))


def get_smiles_cache_path(cache_dir, smiles):
    cache_root = _ensure_cache_root(cache_dir)
    return _cache_file(cache_root, "chembert", get_smiles_cache_key(smiles))


class ArrayFileCache:
    def __init__(self, max_items=512):
        self.max_items = max_items
        self._store = OrderedDict()

    def get(self, path):
        path = str(path)
        if path in self._store:
            value = self._store.pop(path)
            self._store[path] = value
            return value

        value = np.load(path, allow_pickle=False)
        self._store[path] = value
        if len(self._store) > self.max_items:
            self._store.popitem(last=False)
        return value


def _load_cache_vec(cache_root, namespace, key):
    fpath = _cache_file(cache_root, namespace, key)
    if not fpath.exists():
        return None
    try:
        return np.load(fpath, allow_pickle=False)
    except Exception:
        return None


def _save_cache_vec(cache_root, namespace, key, arr):
    fpath = _cache_file(cache_root, namespace, key)
    tmp = fpath.with_suffix(f".tmp.{os.getpid()}.npy")
    np.save(tmp, np.asarray(arr, dtype=np.float32))
    os.replace(tmp, fpath)


# ---------------------------------------------------------------------------
# Protein embeddings — ESM1b (per-token, layer 33)
# ---------------------------------------------------------------------------

def get_esm1b_embeds(sequences, batch_size=8, cache_dir=None, cache_read=True, cache_write=True):
    """
    Returns a dict: {original_sequence_str -> np.float32 array of shape (L, 1280)}
    L = min(len(seq), MAX_PROT_TOK).

    Uses the same ESM1b loader as code/preprocessing/protein_embeddings.py.
    """
    cache_root = _ensure_cache_root(cache_dir)

    # Build unique-sequence map (key = truncated sequence for cache lookup)
    seq_to_truncated = {}
    for seq in sequences:
        seq = str(seq)
        trunc = seq[:MAX_PROT_TOK]
        seq_to_truncated[seq] = trunc

    unique_trunc = list(set(seq_to_truncated.values()))
    trunc_to_embed = {}

    misses = []
    for trunc in unique_trunc:
        if cache_root is not None and cache_read:
            key = _cache_key(PROT_MODEL_ID, trunc)
            cached = _load_cache_vec(cache_root, "esm1b", key)
            if cached is not None:
                trunc_to_embed[trunc] = cached.astype(np.float32)
                continue
        misses.append(trunc)

    if misses:
        try:
            import esm as esm_module
            model, alphabet = esm_module.pretrained.load_model_and_alphabet(PROT_MODEL_ID)
        except ImportError:
            raise ImportError(
                "fair-esm is required. Install with: pip install fair-esm"
            )

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        batch_converter = alphabet.get_batch_converter()

        for start in tqdm(range(0, len(misses), batch_size), desc="ESM1b embedding", unit="batch"):
            batch_seqs = misses[start : start + batch_size]
            # ESM expects (label, seq) pairs
            data = [(str(i), s) for i, s in enumerate(batch_seqs)]
            _, _, batch_tokens = batch_converter(data)

            if torch.cuda.is_available():
                batch_tokens = batch_tokens.cuda()

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)

            representations = results["representations"][33].cpu().numpy()  # [B, L+2, 1280]

            for b_idx, trunc_seq in enumerate(batch_seqs):
                seq_len = min(len(trunc_seq), MAX_PROT_TOK)
                # Slice off BOS/EOS tokens: positions 1 … seq_len (inclusive)
                embed = representations[b_idx, 1 : seq_len + 1].astype(np.float32)
                trunc_to_embed[trunc_seq] = embed
                if cache_root is not None and cache_write:
                    key = _cache_key(PROT_MODEL_ID, trunc_seq)
                    _save_cache_vec(cache_root, "esm1b", key, embed)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Map back to original sequence strings
    result = {}
    for seq in sequences:
        trunc = seq_to_truncated[str(seq)]
        result[str(seq)] = trunc_to_embed[trunc]

    return result


# ---------------------------------------------------------------------------
# SMILES embeddings — ChemBERTa (per-token logits)
# ---------------------------------------------------------------------------

def get_chembert_embeds(smiles_list, batch_size=16, cache_dir=None, cache_read=True, cache_write=True):
    """
    Returns a dict: {smiles_str -> np.float32 array of shape (L, 600)}
    L = actual non-padding token length, ≤ MAX_SMILES_TOK.

    Mirrors get_last_layer_repr() in code/preprocessing/smiles_embeddings.py
    but extracts per-token logits and pools nothing (the model does the pooling
    in MM_TN via attention, not here).
    """
    cache_root = _ensure_cache_root(cache_dir)

    unique_smiles = list(set(str(s) for s in smiles_list))
    smiles_to_embed = {}

    misses = []
    for smi in unique_smiles:
        if cache_root is not None and cache_read:
            key = _cache_key(MOL_MODEL_ID, smi)
            cached = _load_cache_vec(cache_root, "chembert", key)
            if cached is not None:
                smiles_to_embed[smi] = cached.astype(np.float32)
                continue
        misses.append(smi)

    if misses:
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        tokenizer = AutoTokenizer.from_pretrained(MOL_MODEL_ID)
        model = AutoModelForMaskedLM.from_pretrained(MOL_MODEL_ID)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        for start in tqdm(range(0, len(misses), batch_size), desc="ChemBERTa embedding", unit="batch"):
            batch = misses[start : start + batch_size]
            enc = tokenizer(
                batch,
                max_length=MAX_SMILES_TOK,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"]
            attn_mask = enc["attention_mask"]
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attn_mask = attn_mask.cuda()

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
                # logits: [B, L, vocab_size=600]

            logits_np = logits.cpu().numpy()
            attn_np = attn_mask.cpu().numpy()

            for b_idx, smi in enumerate(batch):
                tok_len = int(attn_np[b_idx].sum())
                embed = logits_np[b_idx, :tok_len].astype(np.float32)
                smiles_to_embed[smi] = embed
                if cache_root is not None and cache_write:
                    key = _cache_key(MOL_MODEL_ID, smi)
                    _save_cache_vec(cache_root, "chembert", key, embed)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Map back (handles duplicates in input list)
    result = {}
    for smi in smiles_list:
        result[str(smi)] = smiles_to_embed[str(smi)]

    return result
