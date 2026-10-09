#!/bin/bash
#SBATCH --job-name=spamo-prune-scores
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Computes and caches head / FFN importance scores (pruning.scores_path) for the pruning sweep.
# Usage: sbatch prune_scores.sbatch [ckpt] [config]
# Then:  sbatch --dependency=afterok:<jobid> prune_sweep.sbatch

set -euo pipefail

CKPT="${1:-/pfs/work9/workspace/scratch/ka_uvelk-models/SpaMo/spamo.ckpt}"
CONFIG="${2:-configs/prune.yaml}"
LOG_DIR="/pfs/work9/workspace/scratch/ka_uvelk-models/SpaMo/logs"

source /home/ka/ka_stud/ka_uvelk/miniconda3/etc/profile.d/conda.sh
conda activate SpaMo
cd /pfs/data6/home/ka/ka_stud/ka_uvelk/models/SpaMo

srun python main.py -c "$CONFIG" -e bleu -l "$LOG_DIR" --ckpt "$CKPT" --scores_only
