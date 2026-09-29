#!/bin/bash
#SBATCH --job-name=spamo-eval
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Evaluates a SpaMo checkpoint on the Phoenix14T test split.
# Usage: sbatch eval.sbatch [ckpt] [config]

set -euo pipefail

CKPT="${1:-/pfs/work9/workspace/scratch/ka_uvelk-models/SpaMo/spamo.ckpt}"
CONFIG="${2:-configs/finetune.yaml}"
LOG_DIR="/pfs/work9/workspace/scratch/ka_uvelk-models/SpaMo/logs"

source /home/ka/ka_stud/ka_uvelk/miniconda3/etc/profile.d/conda.sh
conda activate SpaMo
cd /pfs/data6/home/ka/ka_stud/ka_uvelk/models/SpaMo

srun python main.py -c "$CONFIG" -e bleu --train False --test True --ckpt "$CKPT" -l "$LOG_DIR"
