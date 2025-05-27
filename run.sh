#!/bin/bash
#SBATCH -J eval_piqa               
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=8                
#SBATCH --nodes=1                        
#SBATCH --partition=a100-galvani         
#SBATCH --gres=gpu:1                     
#SBATCH --mem=40G                        
#SBATCH --time=1-12:00                   
#SBATCH --output=logs/eval_piqa.%j.out
#SBATCH --error=logs/eval_piqa.%j.err

# Load your Conda setup
source ~/.bashrc
conda activate /home/geiger/gwb345/miniconda/envs/thesis

# Confirm Conda + CUDA‐enabled PyTorch
echo "Using Python: $(which python)"
python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# Paths
CODE_DIR="$WORK/LLMCBench"
MODEL_DIR="/mnt/qb/work/geiger/gwb130/LLMs/hub/models--abacusai--Llama-3-Smaug-8B/snapshots/fe54a7d42160d3d8fcc3289c8c411fd9dd5e8357"

# Use a cache directory you own
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
mkdir -p "$HF_DATASETS_CACHE"

# Run PIQA evaluator
srun python -u $CODE_DIR/evaluate_PIQA.py \
     --path  $MODEL_DIR \
     --split val

