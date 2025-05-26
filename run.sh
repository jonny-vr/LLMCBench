#!/bin/bash
#SBATCH -J eval_mmlu                # Job name
#SBATCH --ntasks=1                  # Single task
#SBATCH --cpus-per-task=8           # Number of CPU cores
#SBATCH --nodes=1                   # All cores on one node
#SBATCH --partition=a100-galvani    # GPU partition
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --mem=40G                   # Total RAM
#SBATCH --time=1-12:00              # D-HH:MM walltime
#SBATCH --output=logs/eval_mmlu.%j.out
#SBATCH --error=logs/eval_mmlu.%j.err


# Load your Conda setup
source ~/.bashrc
conda activate thesis

# Paths
CODE_DIR="$WORK/LLMCBench"
DATA_DIR="$WORK/LLMCBench/data/MMLU"
MODEL_DIR=/mnt/qb/work/geiger/gwb130/LLMs/hub/models--abacusai--Llama-3-Smaug-8B/snapshots/fe54a7d42160d3d8fcc3289c8c411fd9dd5e8357
# Run the evaluator
srun python -u $CODE_DIR/evaluate_mmlu.py \
     --data_dir $DATA_DIR \
     --path $MODEL_DIR \
     --ntrain 0 \
     --seqlen 2048
