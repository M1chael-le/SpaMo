#!/bin/bash
#SBATCH --job-name=spamo-prune
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=72:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

# Pruning sweep: array task i prunes i*10 % of attention heads and FFN neurons (0-80 %),
# tests the pruned model (test_pruned/*), fine-tunes to recover, and tests again (test/*).
# Run prune_scores.sbatch first so all tasks share the cached importance scores:
#   jid=$(sbatch --parsable prune_scores.sbatch)
#   sbatch --dependency=afterok:$jid prune_sweep.sbatch
# Usage: sbatch prune_sweep.sbatch [ckpt] [config]

set -euo pipefail

CKPT="${1:-/pfs/work9/workspace/scratch/ka_uvelk-models/SpaMo/spamo.ckpt}"
CONFIG="${2:-configs/prune.yaml}"
LOG_DIR="/pfs/work9/workspace/scratch/ka_uvelk-models/SpaMo/logs"

PCT=$((SLURM_ARRAY_TASK_ID * 10))
RATIO=$(printf "0.%d" "$SLURM_ARRAY_TASK_ID")

source /home/ka/ka_stud/ka_uvelk/miniconda3/etc/profile.d/conda.sh
conda activate SpaMo
cd /pfs/data6/home/ka/ka_stud/ka_uvelk/models/SpaMo

srun python main.py -c "$CONFIG" -e bleu -l "$LOG_DIR" --ckpt "$CKPT" \
    --head_ratio "$RATIO" --ffn_ratio "$RATIO" -n "prune${PCT}"
