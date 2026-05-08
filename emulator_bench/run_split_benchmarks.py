"""
run_split_benchmarks.py — ProSmith TVT benchmark orchestrator.

Mirrors CataPro emulator_bench/run_split_benchmarks.py exactly in terms of:
  - split_group / threshold discovery logic
  - --value_type (kcat|km|ki|kd|ic50|ec50), --split_groups, --thresholds
  - per-seed runs
  - summary CSV outputs (prosmith_* prefix)
  - hparams_json override
  - dry_run

Feature output dirs: <threshold_dir>/prosmith_features/
Result output dirs:  <threshold_dir>/prosmith_results/seed_<N>/
Summary CSVs:        <value_root>/prosmith_summary_*.csv
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from slurm_utils import DURABLE_TRAINING_FILES, map_to_home, map_to_scratch, sync_selected_files

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "emulator_bench" / "build_tvt_data.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def discover_threshold_dirs(value_root: Path, split_groups, explicit_thresholds=None):
    jobs = []
    for split_group in split_groups:
        split_root = value_root / split_group
        if not split_root.exists():
            continue

        if explicit_thresholds:
            threshold_dirs = [split_root / t for t in explicit_thresholds]
        else:
            threshold_dirs = [
                p for p in sorted(split_root.iterdir())
                if p.is_dir() and p.name.startswith("threshold_")
            ]

        for threshold_dir in threshold_dirs:
            if threshold_dir.exists():
                jobs.append((split_group, threshold_dir.name, threshold_dir))

    return jobs


def ensure_csv_triplet(threshold_dir: Path):
    return (
        threshold_dir / "train.csv",
        threshold_dir / "val.csv",
        threshold_dir / "test.csv",
    )


def _threshold_to_float(name: str):
    try:
        return float(str(name).split("threshold_")[-1])
    except Exception:
        return float("inf")


def _slug(text: str):
    return str(text).replace("/", "_").replace(" ", "_")


def get_split_meta(train_csv, val_csv, test_csv, ratio_tolerance):
    train_size = len(pd.read_csv(train_csv))
    val_size = len(pd.read_csv(val_csv))
    test_size = len(pd.read_csv(test_csv))
    total = train_size + val_size + test_size

    if total == 0:
        train_ratio = val_ratio = test_ratio = 0.0
    else:
        train_ratio = train_size / total
        val_ratio = val_size / total
        test_ratio = test_size / total

    target = (0.8, 0.1, 0.1)
    small_split_flag = int(
        abs(train_ratio - target[0]) > ratio_tolerance
        or abs(val_ratio - target[1]) > ratio_tolerance
        or abs(test_ratio - target[2]) > ratio_tolerance
    )

    return {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "small_split_flag": small_split_flag,
    }


def maybe_build_feature(csv_path, out_pkl, args):
    meta_path = out_pkl.with_suffix(".meta.json")
    if out_pkl.exists() and meta_path.exists() and not args.overwrite:
        return

    cmd = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--input_csv", str(csv_path),
        "--output_pkl", str(out_pkl),
        "--target_col", args.target_col,
        "--sequence_col", args.sequence_col,
        "--smiles_col", args.smiles_col,
        "--prot_batch_size", str(args.prot_batch_size),
        "--mol_batch_size", str(args.mol_batch_size),
        "--cache_dir", args.cache_dir,
    ]
    if args.no_cache_read:
        cmd.append("--no_cache_read")
    if args.no_cache_write:
        cmd.append("--no_cache_write")

    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def run_training(train_pkl, val_pkl, test_pkl, out_dir, args, task_name, seed):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_pkl", str(train_pkl),
        "--val_pkl", str(val_pkl),
        "--test_pkl", str(test_pkl),
        "--out_dir", str(out_dir),
        "--task_name", task_name,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--min_delta", str(args.min_delta),
        "--scheduler", str(args.scheduler),
        "--lr_decay_factor", str(args.lr_decay_factor),
        "--lr_decay_patience", str(args.lr_decay_patience),
        "--min_lr", str(args.min_lr),
        "--lr_warmup_epochs", str(args.lr_warmup_epochs),
        "--lr_warmup_start_factor", str(args.lr_warmup_start_factor),
        "--checkpoint_every", str(args.checkpoint_every),
        "--num_hidden_layers", str(args.num_hidden_layers),
        "--num_gpus", str(args.num_gpus),
        "--device", args.device,
        "--ddp_port", str(args.ddp_port),
        "--seed", str(seed),
        "--sequence_col", args.sequence_col,
        "--smiles_col", args.smiles_col,
        "--num_workers", str(args.num_workers),
        "--prefetch_factor", str(args.prefetch_factor),
        "--cache_items", str(args.cache_items),
        "--grad_accum_steps", str(args.grad_accum_steps),
    ]
    if args.slurm:
        cmd.extend([
            "--slurm",
            "--slurm_sync_dir", str(map_to_home(out_dir, args.slurm_scratch_root, args.slurm_home_root)),
            "--slurm_home_root", args.slurm_home_root,
            "--slurm_scratch_root", args.slurm_scratch_root,
        ])
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _cli_provided_any(*flags):
    argv = sys.argv[1:]
    return any(
        arg == flag or arg.startswith(f"{flag}=")
        for flag in flags
        for arg in argv
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run ProSmith TVT benchmark across split families and thresholds for one value type."
    )
    parser.add_argument(
        "--base_dir",
        default="/home/adhil/github/EMULaToR/data/processed/baselines/ProSmith",
        type=str,
        help="Root directory containing value-type folders such as kcat/km/ki/kd/ic50/ec50.",
    )
    parser.add_argument("--value_type", required=True, choices=["kcat", "km", "ki", "kd", "ic50", "ec50"], type=str)
    parser.add_argument(
        "--split_groups",
        nargs="+",
        default=["enzyme_sequence_splits", "substrate_splits"],
    )
    parser.add_argument("--thresholds", nargs="+", default=None)

    # CSV columns
    parser.add_argument("--target_col", default="log10_value", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)

    # Feature building
    parser.add_argument("--prot_batch_size", default=8, type=int)
    parser.add_argument("--mol_batch_size", default=16, type=int)
    parser.add_argument("--cache_dir", default="emulator_bench/.cache_embeddings", type=str)
    parser.add_argument("--no_cache_read", action="store_true")
    parser.add_argument("--no_cache_write", action="store_true")

    # Training
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--train_batch_size", dest="batch_size", type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--min_delta", default=1e-4, type=float)
    parser.add_argument("--scheduler", default="cosine", choices=["none", "plateau", "cosine"], type=str)
    parser.add_argument("--lr_decay_factor", default=0.5, type=float)
    parser.add_argument("--lr_decay_patience", default=5, type=int)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--lr_warmup_epochs", default=3, type=int)
    parser.add_argument("--lr_warmup_start_factor", default=0.1, type=float)
    parser.add_argument("--checkpoint_every", default=5, type=int)
    parser.add_argument("--num_hidden_layers", default=6, type=int)

    # Device
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_gpus", default=1, type=int,
                        help="Number of GPUs per training job (DDP).")
    parser.add_argument("--ddp_port", default=12557, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--prefetch_factor", default=1, type=int)
    parser.add_argument("--cache_items", default=128, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)

    # HPO override
    parser.add_argument("--hparams_json", type=str, default=None)

    # Run control
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--ratio_tolerance", type=float, default=0.02)
    parser.add_argument(
        "--primary_metric",
        type=str,
        default="MSE",
        choices=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"],
    )
    parser.add_argument("--higher_is_better", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--slurm", action="store_true",
                        help="Run scratch-backed on SLURM and mirror durable outputs back to home.")
    parser.add_argument("--slurm_home_root", default="/home/da24s023", type=str)
    parser.add_argument("--slurm_scratch_root", default="/scratch/da24s023", type=str)

    args = parser.parse_args()

    # Load hparams JSON override
    if args.hparams_json:
        with open(args.hparams_json) as f:
            hp = json.load(f)
        cli_overrides = {
            "batch_size": _cli_provided_any("--batch_size", "--train_batch_size"),
            "lr": _cli_provided_any("--lr"),
            "epochs": _cli_provided_any("--epochs"),
            "patience": _cli_provided_any("--patience"),
            "min_delta": _cli_provided_any("--min_delta"),
            "scheduler": _cli_provided_any("--scheduler"),
            "lr_decay_factor": _cli_provided_any("--lr_decay_factor"),
            "lr_decay_patience": _cli_provided_any("--lr_decay_patience"),
            "min_lr": _cli_provided_any("--min_lr"),
            "lr_warmup_epochs": _cli_provided_any("--lr_warmup_epochs"),
            "lr_warmup_start_factor": _cli_provided_any("--lr_warmup_start_factor"),
            "checkpoint_every": _cli_provided_any("--checkpoint_every"),
            "num_workers": _cli_provided_any("--num_workers"),
            "prefetch_factor": _cli_provided_any("--prefetch_factor"),
            "cache_items": _cli_provided_any("--cache_items"),
            "num_hidden_layers": _cli_provided_any("--num_hidden_layers"),
            "grad_accum_steps": _cli_provided_any("--grad_accum_steps"),
        }
        key_map = [
            ("batch_size", "batch_size", int),
            ("train_batch_size", "batch_size", int),
            ("lr", "lr", float),
            ("epochs", "epochs", int),
            ("patience", "patience", int),
            ("min_delta", "min_delta", float),
            ("scheduler", "scheduler", str),
            ("lr_decay_factor", "lr_decay_factor", float),
            ("lr_decay_patience", "lr_decay_patience", int),
            ("min_lr", "min_lr", float),
            ("lr_warmup_epochs", "lr_warmup_epochs", int),
            ("lr_warmup_start_factor", "lr_warmup_start_factor", float),
            ("checkpoint_every", "checkpoint_every", int),
            ("num_workers", "num_workers", int),
            ("prefetch_factor", "prefetch_factor", int),
            ("cache_items", "cache_items", int),
            ("num_hidden_layers", "num_hidden_layers", int),
            ("grad_accum_steps", "grad_accum_steps", int),
        ]
        for src_key, dest_key, caster in key_map:
            if src_key in hp and not cli_overrides.get(dest_key, False):
                setattr(args, dest_key, caster(hp[src_key]))
        print(f"Loaded hyperparameters from {args.hparams_json}")

    print(
        "Effective training hparams: "
        f"batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}, "
        f"patience={args.patience}, min_delta={args.min_delta}, "
        f"scheduler={args.scheduler}, min_lr={args.min_lr}, warmup_epochs={args.lr_warmup_epochs}, "
        f"checkpoint_every={args.checkpoint_every}, num_hidden_layers={args.num_hidden_layers}, "
        f"grad_accum_steps={args.grad_accum_steps}, "
        f"num_workers={args.num_workers}, prefetch_factor={args.prefetch_factor}, cache_items={args.cache_items}"
    )

    if args.slurm:
        args.base_dir = str(map_to_scratch(args.base_dir, args.slurm_scratch_root, args.slurm_home_root))
        args.cache_dir = str(map_to_scratch(args.cache_dir, args.slurm_scratch_root, args.slurm_home_root))
        print(f"SLURM mode enabled: scratch base_dir={args.base_dir}")
        print(f"SLURM mode enabled: scratch cache_dir={args.cache_dir}")

    base_dir = Path(args.base_dir)
    value_root = base_dir / args.value_type
    home_value_root = map_to_home(value_root, args.slurm_scratch_root, args.slurm_home_root) if args.slurm else value_root

    if not value_root.exists():
        raise FileNotFoundError(f"Value type directory not found: {value_root}")

    jobs = discover_threshold_dirs(value_root, args.split_groups, args.thresholds)
    if not jobs:
        raise RuntimeError("No threshold jobs discovered. Check --base_dir/--value_type/--split_groups/--thresholds")

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not-set>')}")
    print(f"Discovered {len(jobs)} jobs for value_type={args.value_type}")

    if args.dry_run:
        for split_group, threshold_name, threshold_dir in jobs:
            print(f"- {split_group}/{threshold_name}: {threshold_dir}")
        return

    run_rows = []
    progress = tqdm(jobs, desc=f"{args.value_type} benchmark", unit="job")

    for split_group, threshold_name, threshold_dir in progress:
        progress.set_postfix(split=split_group, threshold=threshold_name)

        train_csv, val_csv, test_csv = ensure_csv_triplet(threshold_dir)
        if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
            print(f"[skip] missing csv triplet in {threshold_dir}")
            continue

        split_meta = get_split_meta(train_csv, val_csv, test_csv, args.ratio_tolerance)

        feats_dir = threshold_dir / "prosmith_features"
        out_dir = threshold_dir / "prosmith_results"
        feats_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_pkl = feats_dir / "train_feats.pkl"
        val_pkl = feats_dir / "val_feats.pkl"
        test_pkl = feats_dir / "test_feats.pkl"

        maybe_build_feature(train_csv, train_pkl, args)
        maybe_build_feature(val_csv, val_pkl, args)
        maybe_build_feature(test_csv, test_pkl, args)

        for seed in args.seeds:
            seed_out_dir = out_dir / f"seed_{seed}"
            home_seed_out_dir = map_to_home(seed_out_dir, args.slurm_scratch_root, args.slurm_home_root) if args.slurm else seed_out_dir
            final_test_csv = seed_out_dir / "final_results_test.csv"

            if not final_test_csv.exists() or args.overwrite:
                task_name = f"{args.value_type}_{split_group}_{threshold_name}_seed{seed}"
                run_training(train_pkl, val_pkl, test_pkl, seed_out_dir, args, task_name, seed)
            elif args.slurm:
                sync_selected_files(seed_out_dir, home_seed_out_dir, DURABLE_TRAINING_FILES)

            if final_test_csv.exists():
                row = pd.read_csv(final_test_csv).iloc[0].to_dict()
                row["value_type"] = args.value_type
                row["split_group"] = split_group
                row["threshold"] = threshold_name
                row["seed"] = seed
                row["results_dir"] = str(home_seed_out_dir if args.slurm else seed_out_dir)
                row.update(split_meta)
                run_rows.append(row)

    if not run_rows:
        print("No completed jobs to summarize.")
        return

    runs_df = pd.DataFrame(run_rows)
    runs_df["threshold_num"] = runs_df["threshold"].map(_threshold_to_float)
    runs_df = runs_df.sort_values(["split_group", "threshold_num", "seed"]).drop(columns=["threshold_num"])

    runs_path = value_root / "prosmith_summary_runs.csv"
    runs_df.to_csv(runs_path, index=False)

    for split_group, g_runs in runs_df.groupby("split_group", sort=False):
        split_slug = _slug(split_group)
        runs_df.loc[g_runs.index].to_csv(value_root / f"prosmith_summary_runs__{split_slug}.csv", index=False)

    metric_cols = [c for c in ["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"] if c in runs_df.columns]
    group_cols = ["value_type", "split_group", "threshold"]

    threshold_rows = []
    for keys, g in runs_df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(g["seed"].nunique())
        for c in ["train_size", "val_size", "test_size", "train_ratio", "val_ratio", "test_ratio", "small_split_flag"]:
            row[c] = g[c].iloc[0]
        for m in metric_cols:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_var"] = float(g[m].var(ddof=1)) if len(g) > 1 else 0.0
        threshold_rows.append(row)

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df["threshold_num"] = threshold_df["threshold"].map(_threshold_to_float)
    threshold_df = threshold_df.sort_values(["split_group", "threshold_num"]).drop(columns=["threshold_num"])

    threshold_path = value_root / "prosmith_summary_thresholds.csv"
    threshold_df.to_csv(threshold_path, index=False)

    # Backward-compatible alias
    (value_root / "prosmith_summary.csv").write_bytes((value_root / "prosmith_summary_thresholds.csv").read_bytes())

    for split_group, g_th in threshold_df.groupby("split_group", sort=False):
        split_slug = _slug(split_group)
        g_th.to_csv(value_root / f"prosmith_summary_thresholds__{split_slug}.csv", index=False)
        g_th.to_csv(value_root / f"prosmith_summary__{split_slug}.csv", index=False)

    by_split_rows = []
    for split_group, g in threshold_df.groupby("split_group", sort=False):
        row = {"value_type": args.value_type, "split_group": split_group, "n_thresholds": len(g)}
        for m in metric_cols:
            row[f"{m}_mean_over_thresholds"] = float(g[f"{m}_mean"].mean())
            row[f"{m}_var_over_thresholds"] = float(g[f"{m}_mean"].var(ddof=1)) if len(g) > 1 else 0.0
        by_split_rows.append(row)

    by_split_df = pd.DataFrame(by_split_rows)
    by_split_path = value_root / "prosmith_summary_by_split_group.csv"
    by_split_df.to_csv(by_split_path, index=False)

    metric_key = f"{args.primary_metric}_mean"
    if metric_key in threshold_df.columns:
        ranked_df = threshold_df.sort_values(metric_key, ascending=not args.higher_is_better)
        ranked_df.to_csv(value_root / "prosmith_summary_ranked.csv", index=False)
        for split_group, g_rank in ranked_df.groupby("split_group", sort=False):
            split_slug = _slug(split_group)
            g_rank.to_csv(value_root / f"prosmith_summary_ranked__{split_slug}.csv", index=False)

    if args.slurm:
        home_value_root.mkdir(parents=True, exist_ok=True)
        for src in value_root.glob("prosmith_summary*.csv"):
            sync_selected_files(src.parent, home_value_root, [src.name])

    print(f"Saved runs summary       : {runs_path}")
    print(f"Saved threshold summary  : {threshold_path}")
    print(f"Saved split-group summary: {by_split_path}")


if __name__ == "__main__":
    main()
