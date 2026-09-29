#!/bin/bash
#SBATCH --job-name=spamo-train
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Trains SpaMo from scratch (40 epochs on Phoenix14T take ~3h on one H100).
# Usage: sbatch train.sbatch [config]

set -euo pipefail

CONFIG="${1:-configs/finetune.yaml}"
LOG_DIR="/pfs/work9/workspace/scratch/ka_uvelk-models/SpaMo/logs"

source /home/ka/ka_stud/ka_uvelk/miniconda3/etc/profile.d/conda.sh
conda activate SpaMo
cd /pfs/data6/home/ka/ka_stud/ka_uvelk/models/SpaMo

srun python main.py -c "$CONFIG" -e bleu -l "$LOG_DIR"
