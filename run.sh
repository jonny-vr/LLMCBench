#!/bin/bash
#SBATCH -J eval_arc_e
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=1-12:00
#SBATCH --output=logs/eval_arc_e.%j.out
#SBATCH --error=logs/eval_arc_e.%j.err

source ~/.bashrc
conda activate /home/geiger/gwb345/miniconda/envs/thesis

# Confirm Conda is active
echo "Using Python: $(which python)"
python -c "import torch; print('Torch version:', torch.__version__)"

# Run evaluation
CODE_DIR="$WORK/LLMCBench"
DATA_DIR="$WORK/LLMCBench/data/ARC-C"
MODEL_DIR="/mnt/qb/work/geiger/gwb130/LLMs/hub/models--abacusai--Llama-3-Smaug-8B/snapshots/fe54a7d42160d3d8fcc3289c8c411fd9dd5e8357"

srun python -u $CODE_DIR/evaluate_arc_c.py \
     --data_dir $DATA_DIR \
     --path $MODEL_DIR \
     --ntrain 0 \
     --seqlen 2048

