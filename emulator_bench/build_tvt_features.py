"""
build_tvt_features.py — Build ProSmith TVT split manifests from a raw CSV.

Output pkl layout:
  {
    "df": pd.DataFrame  — rows with columns [sequence_col, smiles_col, label]
    "storage": "lazy_cache_v1"
    "cache_dir": "/abs/path/to/emulator_bench/.cache_embeddings"
    "sequence_col": "sequence"
    "smiles_col": "smiles"
  }

Embeddings are materialized into the persistent on-disk cache and loaded lazily
at training time instead of being duplicated into each split artifact.
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from feature_utils import get_esm1b_embeds, get_chembert_embeds, MAX_PROT_TOK


def _require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _atomic_pickle_dump(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(output_path)


def _write_metadata(output_path, df, n_prot_embeds, n_smiles_embeds, sequence_col, smiles_col, cache_dir):
    meta_path = Path(output_path).with_suffix(".meta.json")
    metadata = {
        "rows": int(len(df)),
        "sequence_col": sequence_col,
        "smiles_col": smiles_col,
        "n_unique_sequences": int(df[sequence_col].nunique()),
        "n_unique_smiles": int(df[smiles_col].nunique()),
        "n_prot_embeds": int(n_prot_embeds),
        "n_smiles_embeds": int(n_smiles_embeds),
        "storage": "lazy_cache_v1",
        "cache_dir": str(Path(cache_dir).resolve()),
    }
    tmp_path = meta_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=2)
    tmp_path.replace(meta_path)


def build_features(
    input_csv,
    output_pkl,
    target_col,
    sequence_col="sequence",
    smiles_col="smiles",
    prot_batch_size=8,
    mol_batch_size=16,
    cache_dir="emulator_bench/.cache_embeddings",
    cache_read=True,
    cache_write=True,
):
    df = pd.read_csv(input_csv)
    _require_columns(df, [sequence_col, smiles_col, target_col])

    # Truncate sequences to ESM1b limit (mirrors create_fasta_file in protein_embeddings.py)
    df[sequence_col] = df[sequence_col].astype(str).str[:MAX_PROT_TOK]
    df[smiles_col] = df[smiles_col].astype(str)

    seq_list = df[sequence_col].tolist()
    smi_list = df[smiles_col].tolist()

    print(f"Unique proteins : {len(set(seq_list))}")
    print(f"Unique SMILES   : {len(set(smi_list))}")
    print(f"Total rows      : {len(df)}")

    prot_embeds = get_esm1b_embeds(
        list(set(seq_list)),
        batch_size=prot_batch_size,
        cache_dir=cache_dir,
        cache_read=cache_read,
        cache_write=cache_write,
    )
    smiles_embeds = get_chembert_embeds(
        list(set(smi_list)),
        batch_size=mol_batch_size,
        cache_dir=cache_dir,
        cache_read=cache_read,
        cache_write=cache_write,
    )

    # Keep only the columns needed for training
    out_df = df[[sequence_col, smiles_col]].copy()
    out_df["label"] = df[target_col].astype(float).values

    payload = {
        "df": out_df,
        "storage": "lazy_cache_v1",
        "cache_dir": str(Path(cache_dir).resolve()),
        "sequence_col": sequence_col,
        "smiles_col": smiles_col,
    }

    out_path = Path(output_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_pickle_dump(payload, out_path)
    _write_metadata(out_path, out_df, len(prot_embeds), len(smiles_embeds), sequence_col, smiles_col, cache_dir)

    print(f"Saved feature manifest: {output_pkl}")
    print(f"  prot_embeds  : {len(prot_embeds)} unique sequences")
    print(f"  smiles_embeds: {len(smiles_embeds)} unique SMILES")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ProSmith TVT feature pickle (ESM1b + ChemBERTa, dict-based)."
    )
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_pkl", required=True, type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)
    parser.add_argument("--target_col", required=True, type=str)
    parser.add_argument("--prot_batch_size", default=8, type=int)
    parser.add_argument("--mol_batch_size", default=16, type=int)
    parser.add_argument(
        "--cache_dir",
        default="emulator_bench/.cache_embeddings",
        type=str,
        help="Directory for persistent embedding cache.",
    )
    parser.add_argument("--no_cache_read", action="store_true")
    parser.add_argument("--no_cache_write", action="store_true")

    args = parser.parse_args()

    build_features(
        input_csv=args.input_csv,
        output_pkl=args.output_pkl,
        target_col=args.target_col,
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
        prot_batch_size=args.prot_batch_size,
        mol_batch_size=args.mol_batch_size,
        cache_dir=args.cache_dir,
        cache_read=not args.no_cache_read,
        cache_write=not args.no_cache_write,
    )
