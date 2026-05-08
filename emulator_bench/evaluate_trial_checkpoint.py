"""
evaluate_trial_checkpoint.py — Evaluate a ProSmith trial checkpoint on train/val/test.

This script loads the best checkpoint from a trial directory, streams inference
split-by-split, writes prediction CSVs incrementally, and computes exact metrics
while keeping RAM bounded. Spearman correlation is computed with external disk
sorts over temporary files instead of storing all predictions in memory.
"""

import argparse
import csv
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "code"))

from training.utils.modules import MM_TN, MM_TNConfig  # noqa: E402
from train_single_target_tvt import ProSmithDataset  # noqa: E402


class RunningRegressionStats:
    def __init__(self):
        self.n = 0
        self.sum_y = 0.0
        self.sum_p = 0.0
        self.sum_yy = 0.0
        self.sum_pp = 0.0
        self.sum_yp = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0

    def update(self, labels, preds):
        for y, p in zip(labels, preds):
            y = float(y)
            p = float(p)
            diff = p - y
            self.n += 1
            self.sum_y += y
            self.sum_p += p
            self.sum_yy += y * y
            self.sum_pp += p * p
            self.sum_yp += y * p
            self.sum_abs += abs(diff)
            self.sum_sq += diff * diff

    def metrics(self):
        if self.n == 0:
            return {
                "PCC": 0.0,
                "SCC": 0.0,
                "R2": 0.0,
                "RMSE": 0.0,
                "MSE": 0.0,
                "MAE": 0.0,
            }

        mse = self.sum_sq / self.n
        rmse = math.sqrt(mse)
        mae = self.sum_abs / self.n

        mean_y = self.sum_y / self.n
        mean_p = self.sum_p / self.n

        var_y = self.sum_yy - self.n * mean_y * mean_y
        var_p = self.sum_pp - self.n * mean_p * mean_p
        cov = self.sum_yp - self.n * mean_y * mean_p

        if var_y <= 0.0 or var_p <= 0.0:
            pcc = 0.0
        else:
            pcc = cov / math.sqrt(var_y * var_p)

        if var_y <= 0.0:
            r2 = 0.0
        else:
            r2 = 1.0 - (self.sum_sq / var_y)

        return {
            "PCC": float(pcc),
            "SCC": 0.0,
            "R2": float(r2),
            "RMSE": float(rmse),
            "MSE": float(mse),
            "MAE": float(mae),
        }


def _build_model(num_hidden_layers=6):
    config = MM_TNConfig.from_dict({
        "s_hidden_size": 600,
        "p_hidden_size": 1280,
        "hidden_size": 768,
        "max_seq_len": 1276,
        "num_hidden_layers": num_hidden_layers,
        "binary_task": False,
    })
    return MM_TN(config)


def _resolve_device(device_str):
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def _load_best_checkpoint(trial_dir, ckpt_path):
    if ckpt_path is not None:
        return Path(ckpt_path)
    best_path = Path(trial_dir) / "bestmodel.pth"
    if best_path.exists():
        return best_path
    raise FileNotFoundError(f"Could not find best checkpoint at {best_path}")


def _run_sort(input_path, output_path, key_specs):
    cmd = ["sort", "-t", "\t", "-o", str(output_path)]
    cmd.extend(key_specs)
    cmd.append(str(input_path))
    subprocess.run(cmd, check=True)


def _write_rank_file(sorted_input, rank_output, value_col_idx):
    current_value = None
    current_rows = []
    next_rank = 1

    with open(sorted_input, "r", newline="") as src, open(rank_output, "w", newline="") as dst:
        reader = csv.reader(src, delimiter="\t")
        writer = csv.writer(dst, delimiter="\t")

        def flush_group(rows, start_rank):
            if not rows:
                return
            end_rank = start_rank + len(rows) - 1
            avg_rank = 0.5 * (start_rank + end_rank)
            for row in rows:
                writer.writerow([row[0], f"{avg_rank:.12f}"])

        for row in reader:
            value = row[value_col_idx]
            if current_value is None:
                current_value = value
                current_rows = [row]
                continue
            if value == current_value:
                current_rows.append(row)
                continue
            flush_group(current_rows, next_rank)
            next_rank += len(current_rows)
            current_value = value
            current_rows = [row]

        flush_group(current_rows, next_rank)


def _pearson_from_rank_files(label_rank_path, pred_rank_path):
    stats = RunningRegressionStats()
    with open(label_rank_path, "r", newline="") as lf, open(pred_rank_path, "r", newline="") as pf:
        label_reader = csv.reader(lf, delimiter="\t")
        pred_reader = csv.reader(pf, delimiter="\t")
        for label_row, pred_row in zip(label_reader, pred_reader):
            if label_row[0] != pred_row[0]:
                raise ValueError("Rank files are misaligned by row id")
            stats.update([float(label_row[1])], [float(pred_row[1])])
    return stats.metrics()["PCC"]


def _compute_spearman_from_disk(raw_tsv_path, work_dir):
    sorted_label = work_dir / "sorted_by_label.tsv"
    sorted_pred = work_dir / "sorted_by_pred.tsv"
    label_ranks = work_dir / "label_ranks.tsv"
    pred_ranks = work_dir / "pred_ranks.tsv"
    label_ranks_by_idx = work_dir / "label_ranks_by_idx.tsv"
    pred_ranks_by_idx = work_dir / "pred_ranks_by_idx.tsv"

    _run_sort(raw_tsv_path, sorted_label, ["-k2,2g", "-k1,1n"])
    _write_rank_file(sorted_label, label_ranks, value_col_idx=1)
    _run_sort(label_ranks, label_ranks_by_idx, ["-k1,1n"])

    _run_sort(raw_tsv_path, sorted_pred, ["-k3,3g", "-k1,1n"])
    _write_rank_file(sorted_pred, pred_ranks, value_col_idx=2)
    _run_sort(pred_ranks, pred_ranks_by_idx, ["-k1,1n"])

    return _pearson_from_rank_files(label_ranks_by_idx, pred_ranks_by_idx)


def _build_loader(input_pkl, batch_size, num_workers, prefetch_factor, cache_items):
    ds = ProSmithDataset(input_pkl, cache_items=cache_items)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": ProSmithDataset.collate_fn,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(ds, **loader_kwargs)
    return ds, loader


def _evaluate_split(model, device, split_name, input_pkl, out_dir, args):
    _, loader = _build_loader(
        input_pkl=input_pkl,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        cache_items=args.cache_items,
    )

    pred_csv_path = Path(out_dir) / f"pred_label_{split_name}.csv"
    metrics_csv_path = Path(out_dir) / f"final_results_{split_name}.csv"

    stats = RunningRegressionStats()
    raw_rows_path = Path(out_dir) / f".tmp_{split_name}_rows.tsv"

    with tempfile.TemporaryDirectory(prefix=f"spearman_{split_name}_", dir=out_dir) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        row_idx = 0

        with open(pred_csv_path, "w", newline="") as pred_f, open(raw_rows_path, "w", newline="") as raw_f:
            pred_writer = csv.writer(pred_f)
            raw_writer = csv.writer(raw_f, delimiter="\t")
            pred_writer.writerow(["pred", "label"])

            model.eval()
            with torch.no_grad():
                for batch in loader:
                    smiles_emb, smiles_attn, protein_emb, protein_attn, labels = [b.to(device) for b in batch]
                    outputs = model(
                        smiles_emb=smiles_emb,
                        smiles_attn=smiles_attn,
                        protein_emb=protein_emb,
                        protein_attn=protein_attn,
                        device=device,
                        gpu=0,
                    )

                    preds = outputs.squeeze(-1).detach().float().cpu().tolist()
                    batch_labels = labels.squeeze(-1).detach().float().cpu().tolist()
                    stats.update(batch_labels, preds)

                    for label, pred in zip(batch_labels, preds):
                        pred_writer.writerow([f"{pred:.6f}", f"{label:.6f}"])
                        raw_writer.writerow([row_idx, f"{label:.12f}", f"{pred:.12f}"])
                        row_idx += 1

        metrics = stats.metrics()
        metrics["SCC"] = _compute_spearman_from_disk(raw_rows_path, tmp_dir)
        pd.DataFrame([metrics], columns=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"]).to_csv(
            metrics_csv_path, index=False
        )

    if raw_rows_path.exists():
        raw_rows_path.unlink()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ProSmith trial checkpoint on train/val/test.")
    parser.add_argument("--trial_dir", required=True, type=str)
    parser.add_argument("--train_pkl", required=True, type=str)
    parser.add_argument("--val_pkl", required=True, type=str)
    parser.add_argument("--test_pkl", required=True, type=str)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--ckpt_path", default=None, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_hidden_layers", default=6, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--prefetch_factor", default=1, type=int)
    parser.add_argument("--cache_items", default=128, type=int)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    trial_dir = Path(args.trial_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else trial_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = _load_best_checkpoint(trial_dir, args.ckpt_path)
    model = _build_model(num_hidden_layers=args.num_hidden_layers)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    split_to_pkl = {
        "train": args.train_pkl,
        "val": args.val_pkl,
        "test": args.test_pkl,
    }
    all_metrics = []
    for split_name, input_pkl in split_to_pkl.items():
        metrics = _evaluate_split(model, device, split_name, input_pkl, out_dir, args)
        all_metrics.append({"split": split_name, **metrics})
        print(
            f"{split_name}: "
            f"PCC={metrics['PCC']:.4f} SCC={metrics['SCC']:.4f} "
            f"R2={metrics['R2']:.4f} RMSE={metrics['RMSE']:.4f} "
            f"MSE={metrics['MSE']:.4f} MAE={metrics['MAE']:.4f}"
        )

    pd.DataFrame(all_metrics).to_csv(out_dir / "final_results_all_splits.csv", index=False)
    pd.DataFrame(
        [{
            "trial_dir": str(trial_dir.resolve()),
            "checkpoint": str(ckpt_path.resolve()),
            "device": str(device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "cache_items": int(args.cache_items),
        }]
    ).to_csv(out_dir / "evaluation_run_summary.csv", index=False)


if __name__ == "__main__":
    if shutil.which("sort") is None:
        raise RuntimeError("This script requires the Unix 'sort' command for disk-backed Spearman computation.")
    main()
