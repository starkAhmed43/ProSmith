"""
tune_optuna.py — Optuna HPO for ProSmith TVT single-target training.

Mirrors CataPro emulator_bench/tune_optuna.py in structure.
Searches over ProSmith-appropriate hyperparameters:
  - train_batch_size  : [4, 8, 12, 16]
  - lr                : 1e-6 .. 1e-4 (log)
  - num_hidden_layers : [2, 4, 6]
  - scheduler/min_lr  : warmup+cosine by default for short runs
  - num_workers / prefetch_factor / cache_items runtime knobs
  - patience          : 5 .. 20
  - min_delta         : 1e-5 .. 1e-3 (log)
  - epochs            : fixed (from --epochs arg)

Writes best hparams JSON + trials CSV under:
  <value_root>/optuna_studies/<study_name>_best_hparams.json
  <value_root>/optuna_studies/<study_name>_trials.csv
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from pathlib import Path

import optuna
import pandas as pd
from tqdm.auto import tqdm

from slurm_utils import map_to_home, map_to_scratch, sync_selected_files, sync_tree

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
    jobs.sort(key=lambda x: (x[0], _threshold_to_float(x[1])))
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


def maybe_build_feature(csv_path, out_pkl, args):
    meta_path = out_pkl.with_suffix(".meta.json")
    if out_pkl.exists() and meta_path.exists() and not args.overwrite_features:
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


def run_training(train_pkl, val_pkl, test_pkl, out_dir, args, seed, hp, device, num_gpus):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_pkl", str(train_pkl),
        "--val_pkl", str(val_pkl),
        "--test_pkl", str(test_pkl),
        "--out_dir", str(out_dir),
        "--task_name", args.task_name,
        "--batch_size", str(hp["train_batch_size"]),
        "--lr", str(hp["lr"]),
        "--epochs", str(hp["epochs"]),
        "--patience", str(hp["patience"]),
        "--min_delta", str(hp["min_delta"]),
        "--scheduler", str(hp["scheduler"]),
        "--lr_decay_factor", str(hp["lr_decay_factor"]),
        "--lr_decay_patience", str(hp["lr_decay_patience"]),
        "--min_lr", str(hp["min_lr"]),
        "--lr_warmup_epochs", str(hp["lr_warmup_epochs"]),
        "--lr_warmup_start_factor", str(hp["lr_warmup_start_factor"]),
        "--checkpoint_every", str(hp["checkpoint_every"]),
        "--num_hidden_layers", str(hp["num_hidden_layers"]),
        "--num_gpus", str(num_gpus),
        "--device", device,
        "--ddp_port", str(args.ddp_port),
        "--seed", str(seed),
        "--sequence_col", args.sequence_col,
        "--smiles_col", args.smiles_col,
        "--num_workers", str(hp["num_workers"]),
        "--prefetch_factor", str(hp["prefetch_factor"]),
        "--cache_items", str(hp["cache_items"]),
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


def _objective_direction(metric_name: str):
    return "maximize" if metric_name in {"PCC", "SCC", "R2"} else "minimize"


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for ProSmith TVT training.")

    parser.add_argument("--base_dir", default="/home/adhil/github/EMULaToR/data/processed/baselines/ProSmith", type=str)
    parser.add_argument("--value_type", required=True, choices=["kcat", "km", "ki", "kd", "ic50", "ec50"], type=str)
    parser.add_argument("--split_groups", nargs="+", default=["enzyme_sequence_splits", "substrate_splits"])
    parser.add_argument("--separate_by_split_group", action="store_true")
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--max_jobs", type=int, default=6)

    parser.add_argument("--target_col", default="log10_value", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)

    parser.add_argument("--prot_batch_size", default=8, type=int)
    parser.add_argument("--mol_batch_size", default=16, type=int)
    parser.add_argument("--cache_dir", default="emulator_bench/.cache_embeddings", type=str)
    parser.add_argument("--no_cache_read", action="store_true")
    parser.add_argument("--no_cache_write", action="store_true")
    parser.add_argument("--overwrite_features", action="store_true")

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--ddp_port", default=12557, type=int)
    parser.add_argument("--num_workers", default=None, type=int,
                        help="If set, keep num_workers fixed; otherwise Optuna tunes it.")
    parser.add_argument("--prefetch_factor", default=None, type=int,
                        help="If set, keep prefetch_factor fixed; otherwise Optuna tunes it when workers > 0.")
    parser.add_argument("--cache_items", default=None, type=int,
                        help="If set, keep cache_items fixed; otherwise Optuna tunes it.")
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--task_name", default="prosmith_optuna", type=str)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])

    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)

    parser.add_argument("--metric", type=str, default="MSE", choices=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--parallel_runs_per_trial", type=int, default=1)
    parser.add_argument("--trial_parallelism", type=int, default=1)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Fix train batch size for all trials; if omitted, Optuna tunes it.")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "plateau", "cosine"])
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_decay_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--lr_warmup_epochs", type=int, default=3)
    parser.add_argument("--lr_warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--slurm", action="store_true",
                        help="Run scratch-backed on SLURM and mirror durable outputs back to home.")
    parser.add_argument("--slurm_home_root", default="/home/da24s023", type=str)
    parser.add_argument("--slurm_scratch_root", default="/scratch/da24s023", type=str)

    args = parser.parse_args()

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
        raise RuntimeError("No threshold jobs discovered.")

    if args.max_jobs and args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not-set>')}")
    print(f"Tuning jobs ({len(jobs)}):")
    for split_group, threshold_name, threshold_dir in jobs:
        print(f"  {split_group}/{threshold_name}: {threshold_dir}")

    if args.dry_run:
        return

    # Build features first
    feature_progress = tqdm(jobs, desc="Preparing features", unit="job")
    prepared_jobs = []
    for split_group, threshold_name, threshold_dir in feature_progress:
        train_csv, val_csv, test_csv = ensure_csv_triplet(threshold_dir)
        if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
            continue

        feats_dir = threshold_dir / "prosmith_features"
        feats_dir.mkdir(parents=True, exist_ok=True)

        train_pkl = feats_dir / "train_feats.pkl"
        val_pkl = feats_dir / "val_feats.pkl"
        test_pkl = feats_dir / "test_feats.pkl"

        maybe_build_feature(train_csv, train_pkl, args)
        maybe_build_feature(val_csv, val_pkl, args)
        maybe_build_feature(test_csv, test_pkl, args)

        prepared_jobs.append((split_group, threshold_name, threshold_dir, train_pkl, val_pkl, test_pkl))

    if not prepared_jobs:
        raise RuntimeError("No valid jobs with train/val/test csv triplets found.")

    base_study_name = args.study_name or f"prosmith_{args.value_type}_{args.metric.lower()}"
    direction = _objective_direction(args.metric)
    artifacts_dir = value_root / "optuna_studies"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    home_artifacts_dir = map_to_home(artifacts_dir, args.slurm_scratch_root, args.slurm_home_root) if args.slurm else artifacts_dir

    def _assigned_device(task_idx):
        if args.devices:
            return args.devices[task_idx % len(args.devices)]
        return args.device

    def _assigned_gpus(task_idx):
        return args.num_gpus

    def run_one_study(study_name, jobs_subset, split_groups_for_metadata):
        sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
        storage = args.storage
        if storage:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                direction=direction,
                sampler=sampler,
            )
            print(f"Optuna storage: {storage}")
        else:
            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                sampler=sampler,
            )
            print("Optuna storage: in-memory")

        run_root = value_root / "prosmith_optuna_runs" / study_name
        run_root.mkdir(parents=True, exist_ok=True)

        def _run_single(task_idx, split_group, threshold_name, train_pkl, val_pkl, test_pkl, trial_number, seed, hp):
            out_dir = run_root / f"trial_{trial_number}" / split_group / threshold_name / f"seed_{seed}"
            metric_csv = out_dir / f"final_results_{args.eval_split}.csv"
            if not metric_csv.exists():
                run_training(
                    train_pkl, val_pkl, test_pkl, out_dir, args, seed, hp,
                    _assigned_device(task_idx), _assigned_gpus(task_idx),
                )
            if not metric_csv.exists():
                raise RuntimeError(f"Missing metrics file for trial {trial_number}: {metric_csv}")
            df = pd.read_csv(metric_csv)
            if args.metric not in df.columns:
                raise RuntimeError(f"Metric {args.metric} not in {metric_csv}")
            return float(df.iloc[0][args.metric])

        def objective(trial: optuna.Trial):
            hp = {
                "train_batch_size": (
                    args.batch_size
                    if args.batch_size is not None
                    else trial.suggest_categorical("train_batch_size", [4, 8, 12, 16])
                ),
                "lr": trial.suggest_float("lr", 1e-6, 1e-4, log=True),
                "num_hidden_layers": trial.suggest_categorical("num_hidden_layers", [2, 4, 6]),
                "scheduler": args.scheduler,
                "lr_decay_factor": args.lr_decay_factor,
                "lr_decay_patience": args.lr_decay_patience,
                "min_lr": (
                    args.min_lr
                    if args.min_lr is not None
                    else trial.suggest_float("min_lr", 1e-7, 1e-5, log=True)
                ),
                "lr_warmup_epochs": args.lr_warmup_epochs,
                "lr_warmup_start_factor": args.lr_warmup_start_factor,
                "checkpoint_every": args.checkpoint_every,
                "num_workers": (
                    args.num_workers
                    if args.num_workers is not None
                    else trial.suggest_categorical("num_workers", [0, 2, 4])
                ),
                "cache_items": (
                    args.cache_items
                    if args.cache_items is not None
                    else trial.suggest_categorical("cache_items", [64, 128, 256, 512])
                ),
                "patience": trial.suggest_int("patience", 5, 20),
                "min_delta": trial.suggest_float("min_delta", 1e-5, 1e-3, log=True),
                "epochs": args.epochs,
            }
            hp["prefetch_factor"] = (
                args.prefetch_factor
                if args.prefetch_factor is not None
                else (trial.suggest_categorical("prefetch_factor", [1, 2, 4]) if hp["num_workers"] > 0 else 1)
            )

            tasks = []
            task_idx = 0
            for split_group, threshold_name, _, train_pkl, val_pkl, test_pkl in jobs_subset:
                for seed in args.seeds:
                    tasks.append((task_idx, split_group, threshold_name, train_pkl, val_pkl, test_pkl, trial.number, seed, hp))
                    task_idx += 1

            metric_values = []
            if args.parallel_runs_per_trial == 1:
                for task in tasks:
                    try:
                        metric_values.append(_run_single(*task))
                    except subprocess.CalledProcessError as e:
                        raise optuna.TrialPruned(f"Training failed: {e}")
                    except Exception as e:
                        raise optuna.TrialPruned(str(e))
            else:
                with ThreadPoolExecutor(max_workers=args.parallel_runs_per_trial) as ex:
                    futures = [ex.submit(_run_single, *task) for task in tasks]
                    for f in as_completed(futures):
                        try:
                            metric_values.append(f.result())
                        except Exception as e:
                            raise optuna.TrialPruned(str(e))

            if not metric_values:
                raise optuna.TrialPruned("No metric values collected.")

            mean_metric = sum(metric_values) / len(metric_values)
            trial.set_user_attr("n_runs", len(metric_values))
            trial.set_user_attr("metric", args.metric)
            trial.set_user_attr("mean_metric", mean_metric)
            return mean_metric

        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.trial_parallelism)

        best_hp = dict(study.best_params)
        if "train_batch_size" not in best_hp:
            best_hp["train_batch_size"] = args.batch_size
        best_hp.setdefault("scheduler", args.scheduler)
        best_hp.setdefault("lr_decay_factor", args.lr_decay_factor)
        best_hp.setdefault("lr_decay_patience", args.lr_decay_patience)
        best_hp.setdefault("min_lr", args.min_lr if args.min_lr is not None else 1e-6)
        best_hp.setdefault("lr_warmup_epochs", args.lr_warmup_epochs)
        best_hp.setdefault("lr_warmup_start_factor", args.lr_warmup_start_factor)
        best_hp.setdefault("checkpoint_every", args.checkpoint_every)
        best_hp.setdefault("num_workers", args.num_workers if args.num_workers is not None else 0)
        best_hp.setdefault("prefetch_factor", args.prefetch_factor if args.prefetch_factor is not None else 1)
        best_hp.setdefault("cache_items", args.cache_items if args.cache_items is not None else 128)
        best_hp["epochs"] = args.epochs

        best_path = artifacts_dir / f"{study_name}_best_hparams.json"
        with open(best_path, "w") as f:
            json.dump(
                {
                    "value_type": args.value_type,
                    "metric": args.metric,
                    "direction": direction,
                    "eval_split": args.eval_split,
                    "seeds": args.seeds,
                    "split_groups": split_groups_for_metadata,
                    "thresholds": args.thresholds,
                    "max_jobs": args.max_jobs,
                    "best_trial_number": study.best_trial.number,
                    "best_value": float(study.best_value),
                    "train_batch_size": int(best_hp["train_batch_size"]),
                    "lr": float(best_hp["lr"]),
                    "num_hidden_layers": int(best_hp["num_hidden_layers"]),
                    "scheduler": str(best_hp["scheduler"]),
                    "lr_decay_factor": float(best_hp["lr_decay_factor"]),
                    "lr_decay_patience": int(best_hp["lr_decay_patience"]),
                    "min_lr": float(best_hp["min_lr"]),
                    "lr_warmup_epochs": int(best_hp["lr_warmup_epochs"]),
                    "lr_warmup_start_factor": float(best_hp["lr_warmup_start_factor"]),
                    "checkpoint_every": int(best_hp["checkpoint_every"]),
                    "num_workers": int(best_hp["num_workers"]),
                    "prefetch_factor": int(best_hp["prefetch_factor"]),
                    "cache_items": int(best_hp["cache_items"]),
                    "patience": int(best_hp["patience"]),
                    "min_delta": float(best_hp["min_delta"]),
                    "epochs": int(best_hp["epochs"]),
                    "grad_accum_steps": int(args.grad_accum_steps),
                },
                f,
                indent=2,
            )

        trials_path = artifacts_dir / f"{study_name}_trials.csv"
        study.trials_dataframe().to_csv(trials_path, index=False)

        if args.slurm:
            sync_selected_files(artifacts_dir, home_artifacts_dir, [best_path.name, trials_path.name])
            best_trial_dir = run_root / f"trial_{study.best_trial.number}"
            home_run_root = map_to_home(run_root, args.slurm_scratch_root, args.slurm_home_root)
            sync_tree(best_trial_dir, home_run_root / f"trial_{study.best_trial.number}")

        print(f"[{study_name}] Best trial: {study.best_trial.number}")
        print(f"[{study_name}] Best {args.metric}: {study.best_value:.6f}")
        print(f"[{study_name}] Hparams saved to: {best_path}")

    if args.separate_by_split_group:
        grouped = {}
        for row in prepared_jobs:
            grouped.setdefault(row[0], []).append(row)
        for split_group, jobs_subset in grouped.items():
            sub_name = f"{base_study_name}__{split_group}"
            print(f"Running separate study for split_group={split_group}: {sub_name}")
            run_one_study(sub_name, jobs_subset, [split_group])
    else:
        run_one_study(base_study_name, prepared_jobs, args.split_groups)


if __name__ == "__main__":
    main()
