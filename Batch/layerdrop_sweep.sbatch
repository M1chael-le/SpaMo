#!/bin/bash
#SBATCH --job-name=spamo-layerdrop
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

# Layer-drop sweep: tasks 0-5 drop 2/4/6/8/10/12 encoder blocks, tasks 6-11 drop
# 2/4/6/8/10/12 decoder blocks (least important first, from pruning.layer_scan).
# Each task tests the cut model (test_pruned/*), fine-tunes to recover, and tests again (test/*).
# Usage: sbatch layerdrop_sweep.sbatch [ckpt] [config] [run name prefix]

set -euo pipefail

CKPT="${1:-/pfs/work9/workspace/scratch/ka_uvelk-models/SpaMo/spamo.ckpt}"
CONFIG="${2:-configs/layerdrop.yaml}"
NAME="${3:-layerdrop}"
LOG_DIR="/pfs/work9/workspace/scratch/ka_uvelk-models/SpaMo/logs"

K=$(( (SLURM_ARRAY_TASK_ID % 6 + 1) * 2 ))
if [ "$SLURM_ARRAY_TASK_ID" -lt 6 ]; then STACK=encoder; else STACK=decoder; fi

source /home/ka/ka_stud/ka_uvelk/miniconda3/etc/profile.d/conda.sh
conda activate SpaMo
cd /pfs/data6/home/ka/ka_stud/ka_uvelk/models/SpaMo

srun python main.py -c "$CONFIG" -e bleu -l "$LOG_DIR" --ckpt "$CKPT" \
    --drop_${STACK} "$K" -n "${NAME}_${STACK:0:3}${K}"
