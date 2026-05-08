#!/bin/bash

# Human-readable name for the job in `squeue`
#SBATCH --job-name=prosmith_kd_bench

# Run on the GPU partition
#SBATCH --partition=gpu

# Request one GPU. Without this, a GPU-partition job may still not get a GPU.
#SBATCH --gres=gpu:1

# Request up to 6 days of walltime. If you omit this, the cluster default may be only 2 hours.
#SBATCH --time=6-00:00:00

# Number of CPU cores to use for data loading and preprocessing
#SBATCH --cpus-per-task=8

# Total RAM available to the job
#SBATCH --mem=120G

# Write stdout/stderr to this file. `%x` expands to job-name and `%j` to job-id.
#SBATCH --output=/home/da24s023/slurm_logs/%x_%j.out

# Stop immediately on errors:
# -e: exit on command failure
# -u: error on unset variable
# -o pipefail: fail if any command in a pipeline fails
set -euo pipefail

# Create the log directory if it does not already exist
mkdir -p /home/da24s023/slurm_logs

# Move into the repository root so relative paths behave predictably
cd /home/da24s023/github/ProSmith

# Load the cluster-provided conda module
module load anaconda3

# Enable `conda activate` in this non-interactive shell
source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh

# Activate the ProSmith environment from shared home storage
conda activate /home/da24s023/.conda/envs/prosmith

# Log the resolved interpreter for debugging
which python

# Launch a benchmark run.
# `--slurm` enables scratch-backed behavior:
# - heavy intermediates and checkpoints go under /scratch/da24s023
# - final model/results are mirrored back under /home/da24s023
CUDA_VISIBLE_DEVICES=0 python /home/da24s023/github/ProSmith/emulator_bench/run_split_benchmarks.py \
  --base_dir /home/da24s023/github/EMULaToR/data/processed/baselines/ProSmith \
  --value_type kd \
  --split_groups enzyme_sequence_splits substrate_splits \
  --thresholds threshold_0.08 \
  --seeds 42 \
  --batch_size 32 \
  --device cuda:0 \
  --slurm
