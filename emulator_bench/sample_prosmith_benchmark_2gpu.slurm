#!/bin/bash

# Human-readable name for the job in `squeue`
#SBATCH --job-name=prosmith_kd_bench_2gpu

# Run on the GPU partition
#SBATCH --partition=gpu

# Request both GPUs on the node
#SBATCH --gres=gpu:2

# Request up to 6 days of walltime
#SBATCH --time=6-00:00:00

# Number of CPU cores to use for data loading and preprocessing
#SBATCH --cpus-per-task=12

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

# Make both allocated GPUs visible to PyTorch.
# `--num_gpus 2` tells ProSmith's emulator_bench trainer to use DDP across them.
# `--batch_size` is per GPU, so effective batch size here is 32 * 2 = 64
# before any gradient accumulation.
CUDA_VISIBLE_DEVICES=0,1 python /home/da24s023/github/ProSmith/emulator_bench/run_split_benchmarks.py \
  --base_dir /home/da24s023/github/EMULaToR/data/processed/baselines/ProSmith \
  --value_type kd \
  --split_groups enzyme_sequence_splits substrate_splits \
  --thresholds threshold_0.08 \
  --seeds 42 \
  --batch_size 32 \
  --num_gpus 2 \
  --device cuda:0 \
  --slurm
