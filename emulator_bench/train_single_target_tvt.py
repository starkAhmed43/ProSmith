"""
train_single_target_tvt.py — ProSmith TVT trainer for kcat/Km/Ki.

Supports single-GPU and multi-GPU (DDP via mp.spawn).
Uses MM_TN from code/training/utils/modules.py — no model code is duplicated here.

Usage (single GPU):
  python emulator_bench/train_single_target_tvt.py \
    --train_pkl .../train_feats.pkl \
    --val_pkl   .../val_feats.pkl   \
    --test_pkl  .../test_feats.pkl  \
    --out_dir   .../results         \
    --device cuda:0

Usage (multi-GPU, 2 GPUs):
  CUDA_VISIBLE_DEVICES=0,1 python emulator_bench/train_single_target_tvt.py \
    --train_pkl .../train_feats.pkl \
    --val_pkl   .../val_feats.pkl   \
    --test_pkl  .../test_feats.pkl  \
    --out_dir   .../results         \
    --num_gpus 2
"""

import argparse
import datetime
import json
import os
import pickle
import random
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Resolve ProSmith module path without colliding with emulator_bench/utils.py
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent  # ProSmith/
sys.path.insert(0, str(_REPO_ROOT / "code"))

from training.utils.modules import MM_TN, MM_TNConfig  # noqa: E402

from utils import (  # noqa: E402  (emulator_bench/utils.py)
    EarlyStopping,
    evaluate,
    out_results,
    run_a_training_epoch,
    run_eval_mse_epoch,
    run_an_eval_epoch,
    write_logfile,
)
from feature_utils import ArrayFileCache, get_protein_cache_path, get_smiles_cache_path  # noqa: E402
from slurm_utils import DURABLE_TRAINING_FILES, sync_selected_files  # noqa: E402

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


_INTERRUPT_REQUESTED = False


def _request_interrupt(signum, _frame):
    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = True
    print(f"Received signal {signum}. Will checkpoint and stop after the current epoch.")


def _install_signal_handlers():
    for sig_name in ["SIGTERM", "SIGUSR1", "SIGINT"]:
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _request_interrupt)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProSmithDataset(Dataset):
    """
    Reads a split manifest produced by build_tvt_features.py.
    Supports both:
      1. legacy eager payloads with in-file prot_embeds/smiles_embeds
      2. lazy cache manifests that load embeddings from disk on demand
    """

    MAX_PROT = 1018
    MAX_SMILES = 256

    def __init__(self, pkl_path, sequence_col="sequence", smiles_col="smiles", shuffle=False, seed=42, cache_items=128):
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        self.df = payload["df"].reset_index(drop=True)
        self.prot_embeds = payload.get("prot_embeds")
        self.smiles_embeds = payload.get("smiles_embeds")
        self.storage = payload.get("storage", "legacy_eager_v1")
        self.cache_dir = payload.get("cache_dir")
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.cache_items = cache_items
        self._array_cache = None

        if shuffle:
            self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.sequences = self.df[self.sequence_col].astype(str).tolist()
        self.smiles = self.df[self.smiles_col].astype(str).tolist()
        self.labels = self.df["label"].astype(float).tolist()
        self.length_keys = [
            (
                min(len(seq), self.MAX_PROT),
                min(len(smi), self.MAX_SMILES),
            )
            for seq, smi in zip(self.sequences, self.smiles)
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        smi = self.smiles[idx]
        label = self.labels[idx]

        if self.storage == "lazy_cache_v1":
            if self.cache_dir is None:
                raise ValueError("Lazy cache manifest missing cache_dir")
            if self._array_cache is None:
                self._array_cache = ArrayFileCache(max_items=self.cache_items)
            prot_arr = self._array_cache.get(get_protein_cache_path(self.cache_dir, seq))
            smiles_arr = self._array_cache.get(get_smiles_cache_path(self.cache_dir, smi))
        else:
            prot_arr = self.prot_embeds[seq]    # [L_p, 1280]
            smiles_arr = self.smiles_embeds[smi]  # [L_s, 600]

        if smiles_arr.shape[0] > self.MAX_SMILES:
            smiles_arr = smiles_arr[: self.MAX_SMILES]

        return prot_arr, smiles_arr, label

    @staticmethod
    def collate_fn(batch):
        """Pad protein and SMILES tensors to batch maximums."""
        prot_arrs, smiles_arrs, labels = zip(*batch)

        max_prot = max(a.shape[0] for a in prot_arrs)
        max_smiles = max(a.shape[0] for a in smiles_arrs)

        prot_dim = prot_arrs[0].shape[1]    # 1280
        smiles_dim = smiles_arrs[0].shape[1]  # 600

        B = len(batch)
        prot_padded = np.zeros((B, max_prot, prot_dim), dtype=np.float32)
        prot_mask = np.zeros((B, max_prot), dtype=np.float32)
        smiles_padded = np.zeros((B, max_smiles, smiles_dim), dtype=np.float32)
        smiles_mask = np.zeros((B, max_smiles), dtype=np.float32)

        for i, (p, s) in enumerate(zip(prot_arrs, smiles_arrs)):
            prot_padded[i, : p.shape[0]] = p
            prot_mask[i, : p.shape[0]] = 1.0
            smiles_padded[i, : s.shape[0]] = s
            smiles_mask[i, : s.shape[0]] = 1.0

        labels_t = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # [B, 1]

        return (
            torch.from_numpy(smiles_padded),   # [B, max_smiles, 600]
            torch.from_numpy(smiles_mask),     # [B, max_smiles]
            torch.from_numpy(prot_padded),     # [B, max_prot, 1280]
            torch.from_numpy(prot_mask),       # [B, max_prot]
            labels_t,
        )


class LengthBucketBatchSampler(Sampler):
    """
    Batch indices with similar protein/SMILES lengths together to reduce padding.
    Uses an upstream sampler so DDP sharding and epoch seeding still work.
    """

    def __init__(self, sampler, length_keys, batch_size, bucket_multiplier=50, drop_last=False):
        self.sampler = sampler
        self.length_keys = length_keys
        self.batch_size = batch_size
        self.bucket_size = max(batch_size, batch_size * bucket_multiplier)
        self.drop_last = drop_last

    def __iter__(self):
        pool = []
        rng = random.Random()

        def emit_batches(indices):
            sorted_indices = sorted(
                indices,
                key=lambda idx: self.length_keys[idx][0] + self.length_keys[idx][1],
            )
            batches = [
                sorted_indices[i: i + self.batch_size]
                for i in range(0, len(sorted_indices), self.batch_size)
            ]
            if self.drop_last and batches and len(batches[-1]) < self.batch_size:
                batches = batches[:-1]
            rng.shuffle(batches)
            for batch in batches:
                yield batch

        for idx in self.sampler:
            pool.append(idx)
            if len(pool) >= self.bucket_size:
                yield from emit_batches(pool)
                pool = []

        if pool:
            yield from emit_batches(pool)

    def __len__(self):
        sampler_len = len(self.sampler)
        if self.drop_last:
            return sampler_len // self.batch_size
        return (sampler_len + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _setup_ddp(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def _cleanup_ddp():
    dist.destroy_process_group()


def _is_main(rank):
    return rank == 0


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def _build_model(num_hidden_layers, binary_task=False):
    config = MM_TNConfig.from_dict({
        "s_hidden_size": 600,
        "p_hidden_size": 1280,
        "hidden_size": 768,
        "max_seq_len": 1276,
        "num_hidden_layers": num_hidden_layers,
        "binary_task": binary_task,
    })
    return MM_TN(config)


def _resolve_mixed_precision(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, "fp32", None
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    if major >= 8:
        return torch.bfloat16, "bf16-mixed", device_index
    return torch.float16, "fp16-mixed", device_index


def _checkpoint_paths(out_dir):
    out_dir = Path(out_dir)
    return {
        "latest": out_dir / "checkpoint_last.pt",
        "best": out_dir / "bestmodel.pth",
        "state": out_dir / "run_state.json",
    }


def _build_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_decay_factor,
            patience=args.lr_decay_patience,
            min_lr=args.min_lr,
        )
    if args.scheduler == "cosine":
        warmup_epochs = max(0, min(args.lr_warmup_epochs, args.epochs - 1))
        cosine_epochs = max(1, args.epochs - warmup_epochs)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.lr_warmup_start_factor,
            end_factor=1.0,
            total_iters=max(1, warmup_epochs),
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=args.min_lr,
        )
        if warmup_epochs == 0:
            return cosine
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def _save_json(path, payload):
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _save_checkpoint(path, model, optimizer, scheduler, stopper, scaler, epoch, best_val_loss, start_time, history, multi_gpu, precision_mode):
    state = {
        "epoch": int(epoch),
        "model_state_dict": (model.module.state_dict() if multi_gpu else model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
        "stopper_state": stopper.state_dict(),
        "scaler_state_dict": (scaler.state_dict() if scaler is not None and scaler.is_enabled() else None),
        "best_val_loss": float(best_val_loss),
        "start_time": start_time,
        "history": history,
        "precision_mode": precision_mode,
    }
    tmp_path = Path(str(path) + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def _load_checkpoint(path, model, optimizer, scheduler, stopper, scaler, device, multi_gpu):
    ckpt = torch.load(path, map_location=device)
    target_model = model.module if multi_gpu else model
    target_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler_state = ckpt.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    stopper.load_state_dict(ckpt.get("stopper_state", {}))
    scaler_state = ckpt.get("scaler_state_dict")
    if scaler is not None and scaler.is_enabled() and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    return ckpt


# ---------------------------------------------------------------------------
# Training worker (runs on each process / GPU)
# ---------------------------------------------------------------------------

def _train_worker(rank, world_size, args):
    _install_signal_handlers()
    set_seed(args.seed + rank)

    # ---- Device & DDP setup ----
    multi_gpu = world_size > 1
    if multi_gpu:
        _setup_ddp(rank, world_size, args.ddp_port)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
    else:
        if args.device.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            device = torch.device("cpu")

    autocast_dtype, precision_mode, precision_device_index = _resolve_mixed_precision(device)

    out_dir = Path(args.out_dir)
    if _is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Datasets & Loaders ----
    train_ds = ProSmithDataset(
        args.train_pkl,
        args.sequence_col,
        args.smiles_col,
        shuffle=True,
        seed=args.seed,
        cache_items=args.cache_items,
    )
    val_ds = ProSmithDataset(args.val_pkl, args.sequence_col, args.smiles_col, cache_items=args.cache_items)
    test_ds = ProSmithDataset(args.test_pkl, args.sequence_col, args.smiles_col, cache_items=args.cache_items)

    if multi_gpu:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    else:
        train_sampler = None

    loader_kwargs = dict(
        collate_fn=ProSmithDataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    if args.smart_batching:
        if train_sampler is None:
            train_sampler = torch.utils.data.RandomSampler(train_ds)
        train_batch_sampler = LengthBucketBatchSampler(
            train_sampler,
            train_ds.length_keys,
            batch_size=args.batch_size,
            bucket_multiplier=args.bucket_multiplier,
            drop_last=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            **loader_kwargs,
        )
    eval_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # ---- Model ----
    model = _build_model(num_hidden_layers=args.num_hidden_layers)
    model = model.to(device)

    if multi_gpu:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = _build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype == torch.float16))
    stopper = EarlyStopping(args.patience, args.min_delta)

    ckpt_paths = _checkpoint_paths(out_dir)
    best_model_path = ckpt_paths["best"]
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_val_loss = float("inf")
    start_epoch = 0
    record_data = []

    if args.resume and ckpt_paths["latest"].exists():
        ckpt = _load_checkpoint(ckpt_paths["latest"], model, optimizer, scheduler, stopper, scaler, device, multi_gpu)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        start_time = ckpt.get("start_time", start_time)
        record_data = list(ckpt.get("history", []))
        if _is_main(rank):
            print(f"Resuming from checkpoint: {ckpt_paths['latest']} (epoch {start_epoch})")

    if start_epoch >= args.epochs:
        if _is_main(rank):
            print(f"Checkpoint already reached requested epochs ({args.epochs}). Re-running final evaluation only.")

    if _is_main(rank):
        if precision_device_index is not None:
            gpu_name = torch.cuda.get_device_name(precision_device_index)
            major, minor = torch.cuda.get_device_capability(precision_device_index)
            print(f"CUDA device: {gpu_name} | compute capability: {major}.{minor}")
        print(f"Mixed precision mode: {precision_mode}")
        print(
            f"Scheduler: {args.scheduler} | min_lr: {args.min_lr:.2e} | "
            f"warmup_epochs: {args.lr_warmup_epochs} | warmup_start_factor: {args.lr_warmup_start_factor:.2f}"
        )

    epoch_bar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="epoch", disable=(not _is_main(rank)))

    for epoch in epoch_bar:
        if multi_gpu and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_eval = run_a_training_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
            rank=rank,
            progress_desc=f"Epoch {epoch + 1}/{args.epochs}",
            progress_disable=(not _is_main(rank)),
            grad_accum_steps=args.grad_accum_steps,
        )
        if multi_gpu:
            dist.barrier()

        if _is_main(rank):
            val_loader = DataLoader(val_ds, **eval_loader_kwargs)
            val_mse_eval = run_eval_mse_epoch(
                model.module if multi_gpu else model,
                val_loader,
                device,
                rank=rank,
                autocast_dtype=autocast_dtype,
            )
        else:
            val_mse_eval = np.zeros(2, dtype=np.float64)

        if _is_main(rank):
            epoch_bar.set_postfix(
                train_r2=f"{train_eval[2]:.4f}",
                train_mse=f"{train_eval[4]:.4f}",
                val_mse=f"{val_mse_eval[0]:.4f}",
                val_loss=f"{val_mse_eval[1]:.4f}",
            )

            val_eval_row = np.array(
                [np.nan, np.nan, np.nan, np.nan, val_mse_eval[0], np.nan, val_mse_eval[1]],
                dtype=np.float64,
            )
            record_data.append(np.concatenate([np.array([epoch]), train_eval, val_eval_row]))
            write_logfile(epoch, record_data, str(out_dir / "logfile.csv"))
            best_val_loss = min(best_val_loss, float(val_mse_eval[1]))
            current_lr = float(optimizer.param_groups[0]["lr"])

        if multi_gpu:
            scheduler_metric_t = torch.tensor(float(val_mse_eval[1]) if _is_main(rank) else 0.0, device=device)
            dist.broadcast(scheduler_metric_t, src=0)
            scheduler_metric = float(scheduler_metric_t.item())
        else:
            scheduler_metric = float(val_mse_eval[1]) if _is_main(rank) else 0.0

        if _is_main(rank):
            is_best, stop = stopper.check(epoch, float(val_mse_eval[1]))
        else:
            is_best, stop = False, False

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(scheduler_metric)
            else:
                scheduler.step()

        if _is_main(rank) and (_INTERRUPT_REQUESTED or stop):
            _save_checkpoint(
                ckpt_paths["latest"],
                model,
                optimizer,
                scheduler,
                stopper,
                scaler,
                epoch,
                best_val_loss,
                start_time,
                record_data,
                multi_gpu,
                precision_mode,
            )
            _save_json(
                ckpt_paths["state"],
                {
                    "task_name": args.task_name,
                    "epoch_completed": int(epoch),
                    "epochs_requested": int(args.epochs),
                    "best_val_loss": float(best_val_loss),
                    "checkpoint_last": str(ckpt_paths["latest"]),
                    "checkpoint_best": str(best_model_path),
                    "precision_mode": precision_mode,
                    "scheduler": args.scheduler,
                    "lr": current_lr,
                },
            )

        if is_best and _is_main(rank):
            state = model.module.state_dict() if multi_gpu else model.state_dict()
            torch.save(state, best_model_path)
            if args.slurm and args.slurm_sync_dir:
                sync_selected_files(out_dir, args.slurm_sync_dir, ["bestmodel.pth", "run_state.json", "logfile.csv"])

        # Broadcast stop signal from rank 0 to all ranks
        if multi_gpu:
            stop_t = torch.tensor(int(stop), device=device)
            dist.broadcast(stop_t, src=0)
            stop = bool(stop_t.item())

        if _INTERRUPT_REQUESTED:
            stop = True

        if stop:
            if _is_main(rank):
                epoch_bar.write("Early stopping triggered.")
            break

    # ---- Final evaluation (rank 0 only) ----
    if _is_main(rank):
        _save_checkpoint(
            ckpt_paths["latest"],
            model,
            optimizer,
            scheduler,
            stopper,
            scaler,
            max(start_epoch, len(record_data) - 1),
            best_val_loss,
            start_time,
            record_data,
            multi_gpu,
            precision_mode,
        )
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(out_dir / "time_running.dat", "w") as f:
            f.write(f"Start Time:  {start_time}\n")
            f.write(f"End Time:  {end_time}\n")

        if best_model_path.exists():
            eval_model = _build_model(num_hidden_layers=args.num_hidden_layers)
            eval_model.load_state_dict(torch.load(best_model_path, map_location=device))
            eval_model = eval_model.to(device)
            eval_model.eval()
        else:
            eval_model = model.module if multi_gpu else model

        # Use fresh (non-distributed) loaders for final eval so indices are clean
        val_loader_full = DataLoader(val_ds, **eval_loader_kwargs)
        test_loader_full = DataLoader(test_ds, **eval_loader_kwargs)

        val_pred, val_label, val_eval = run_an_eval_epoch(eval_model, val_loader_full, device, autocast_dtype=autocast_dtype)
        test_pred, test_label, test_eval = run_an_eval_epoch(eval_model, test_loader_full, device, autocast_dtype=autocast_dtype)

        out_results(val_eval, str(out_dir / "results_val.csv"))
        out_results(test_eval, str(out_dir / "results_test.csv"))

        val_df = pd.DataFrame(
            np.stack([val_pred, val_label], axis=1),
            columns=["pred", "label"],
        )
        test_df = pd.DataFrame(
            np.stack([test_pred, test_label], axis=1),
            columns=["pred", "label"],
        )
        val_df.to_csv(out_dir / "pred_label_val.csv", float_format="%.4f", index=False)
        test_df.to_csv(out_dir / "pred_label_test.csv", float_format="%.4f", index=False)

        val_metrics = pd.DataFrame(
            np.array([evaluate(val_df["label"].values, val_df["pred"].values)]),
            columns=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"],
        )
        val_metrics.to_csv(out_dir / "final_results_val.csv", index=False)

        test_metrics = pd.DataFrame(
            np.array([evaluate(test_df["label"].values, test_df["pred"].values)]),
            columns=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"],
        )
        test_metrics.to_csv(out_dir / "final_results_test.csv", index=False)

        summary = pd.DataFrame({
            "task_name": [args.task_name],
            "train_size": [len(train_ds)],
            "val_size": [len(val_ds)],
            "test_size": [len(test_ds)],
            "checkpoint": [str(best_model_path)],
            "checkpoint_last": [str(ckpt_paths["latest"])],
            "precision_mode": [precision_mode],
            "scheduler": [args.scheduler],
            "final_lr": [float(optimizer.param_groups[0]["lr"])],
        })
        summary.to_csv(out_dir / "run_summary.csv", index=False)
        _save_json(
            ckpt_paths["state"],
            {
                "task_name": args.task_name,
                "status": "completed",
                "epochs_requested": int(args.epochs),
                "checkpoint_last": str(ckpt_paths["latest"]),
                "checkpoint_best": str(best_model_path),
                "precision_mode": precision_mode,
                "scheduler": args.scheduler,
                "final_lr": float(optimizer.param_groups[0]["lr"]),
            },
        )
        if args.slurm and args.slurm_sync_dir:
            sync_selected_files(out_dir, args.slurm_sync_dir, DURABLE_TRAINING_FILES)

        print(f"Done. Results saved to {out_dir}")

    if multi_gpu:
        _cleanup_ddp()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    world_size = args.num_gpus if args.num_gpus > 0 else 1

    if world_size > 1 and torch.cuda.is_available():
        mp.spawn(_train_worker, nprocs=world_size, args=(world_size, args))
    else:
        _train_worker(0, 1, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProSmith TVT trainer (MM_TN, kcat/Km/Ki).")
    parser.add_argument("--train_pkl", required=True, type=str)
    parser.add_argument("--val_pkl", required=True, type=str)
    parser.add_argument("--test_pkl", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--task_name", default="single_target", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)

    # MM_TN architecture
    parser.add_argument("--num_hidden_layers", type=int, default=6)

    # Training
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Accumulate gradients over this many microbatches before each optimizer step.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "plateau", "cosine"])
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_decay_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_warmup_epochs", type=int, default=3)
    parser.add_argument("--lr_warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for single-GPU runs (ignored when --num_gpus > 1).")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for DDP. 1 = single GPU, >1 = multi-GPU DDP.")
    parser.add_argument("--ddp_port", type=int, default=12557,
                        help="TCP port for DDP rendezvous.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--cache_items", type=int, default=128,
                        help="Per-worker LRU size for lazy embedding arrays.")
    parser.add_argument("--smart_batching", dest="smart_batching", action="store_true",
                        help="Group similar sequence/SMILES lengths into the same training batches.")
    parser.add_argument("--no_smart_batching", dest="smart_batching", action="store_false")
    parser.set_defaults(smart_batching=True)
    parser.add_argument("--bucket_multiplier", type=int, default=50,
                        help="Pool size multiplier for smart batching. Larger pools reduce padding more.")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--slurm_sync_dir", default=None, type=str)
    parser.add_argument("--slurm_home_root", default="/home/da24s023", type=str)
    parser.add_argument("--slurm_scratch_root", default="/scratch/da24s023", type=str)

    main(parser.parse_args())
