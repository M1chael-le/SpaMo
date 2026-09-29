#!/bin/bash
#SBATCH --job-name=spamo-csl-vit
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Extracts CLIP ViT-L/14 spatial features (S2, scales 1 2) for CSL-Daily.
# The script does not resume: a rerun starts over from the dev split.
# Usage: sbatch Batch/csl_extract_vit.sbatch [anno_root] [save_dir]

set -euo pipefail

ANNO_ROOT="${1:-./preprocess/CSL-Daily}"
SAVE_DIR="${2:-/pfs/work9/workspace/scratch/ka_uvelk-models/SpaMo}"
VIDEO_ROOT="/pfs/work9/workspace/scratch/ka_uvelk-datasets/CSL-Daily"
CACHE_DIR="/pfs/work9/workspace/scratch/ka_uvelk-models/SpaMo/hf_cache"

# Some from_pretrained calls in the script ignore --cache_dir
export HF_HUB_CACHE="$CACHE_DIR"

source /home/ka/ka_stud/ka_uvelk/miniconda3/etc/profile.d/conda.sh
conda activate SpaMo
cd /pfs/data6/home/ka/ka_stud/ka_uvelk/models/SpaMo

srun python scripts/vit_extract_feature.py \
    --anno_root "$ANNO_ROOT" \
    --model_name openai/clip-vit-large-patch14 \
    --video_root "$VIDEO_ROOT" \
    --cache_dir "$CACHE_DIR" \
    --save_dir "$SAVE_DIR" \
    --s2_mode s2wrapping \
    --scales 1 2 \
    --batch_size 32 \
    --device cuda:0
