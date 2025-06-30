#!/bin/bash
#SBATCH -J eval_wiki2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:2
#SBATCH --mem=40G
#SBATCH --time=0-12:00
#SBATCH --output=logs/eval_wiki2.%j.out
#SBATCH --error=logs/eval_wiki2.%j.err

# HOME="/home/geiger/gwb345"
# echo "===== DEBUG START ====="
# echo "This job’s HOME = $HOME"
# echo "Listing cond a.sh under the absolute path:"
# ls -ld /home/geiger/gwb345/miniconda/etc/profile.d/conda.sh \
#    && echo "→ Found it!" \
#    || echo "→ Not found on compute node!"
# echo "Listing your $HOME tree a bit:"
# ls -l $HOME || echo "Cannot ls $HOME"
# echo "===== DEBUG END ====="

# -------- Conda & HF setup --------
# export PATH="/home/geiger/gwb345/miniconda/bin:$PATH"
# source "/home/geiger/gwb345/miniconda/etc/profile.d/conda.sh"
export PATH="/mnt/lustre/work/geiger/gwb345/miniconda/bin:$PATH"
source "/mnt/lustre/work/geiger/gwb345/miniconda/etc/profile.d/conda.sh"
conda activate thesis

export HF_HUB_TOKEN="xxx"  # Replace with your actual Hugging Face token
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export CODE_DIR="/mnt/lustre/work/geiger/gwb345/LLMCBench"

LLAMA_2_7B="/mnt/lustre/work/geiger/gwb345/models/llama-2/llama-2-7b-hf"
LLAMA_2_13B="/mnt/lustre/work/geiger/gwb345/models/llama-2/llama-2-13b-hf"
LLAMA_2_70B="/mnt/lustre/work/geiger/gwb345/models/llama-2/llama-2-70b-hf"
LLAMA_3_8B="/mnt/lustre/work/geiger/gwb345/models/llama-3/llama-3.1-8b-hf"
LLAMA_3_70B="/mnt/lustre/work/geiger/gwb345/models/llama-3/Llama-3.1-70B"
LLAMA_4_SCOUT_109B_17B="/mnt/lustre/work/geiger/gwb345/models/llama-4/Llama-4-Scout-17B-16E"
GEMMA_3_1B="/mnt/lustre/work/geiger/gwb345/models/gemma-3/gemma-3-1b-pt"
GEMMA_3_4B="/mnt/lustre/work/geiger/gwb345/models/gemma-3/gemma-3-4b-pt"
GEMMA_3_12B="/mnt/lustre/work/geiger/gwb345/models/gemma-3/gemma-3-12b-pt"
GEMMA_3_27B="/mnt/lustre/work/geiger/gwb345/models/gemma-3/gemma-3-27b-pt"
QWEN_3_0_6_BASE="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-0.6B-Base"
QWEN_3_1_7B_BASE="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-1.7B-Base"
QWEN_3_4B_BASE="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-4B-Base"
QWEN_3_8B_BASE="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-8B-Base"
QWEN_3_14B_BASE="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-14B-Base"
QWEN_3_32B="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-32B"
QWEN_QWQ="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen-QwQ-32B"
Qwen3_30B_A3B="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-30B-A3B-Base"
DEEPSEEK_MOE_16B_BASE="/mnt/lustre/work/geiger/gwb345/models/deepseek/deepseek-moe-16b-base"
DEEPSEEK_R1_LLAMA_3_8B="/mnt/lustre/work/geiger/gwb345/models/deepseek/DeepSeek-R1-Distill-Llama-8B"
DEEPSEEK_R1_LLAMA_3_70B="/mnt/lustre/work/geiger/gwb345/models/deepseek/DeepSeek-R1-Distill-Llama-70B"

######## create offload model dirs ###########

export JOB_OFFLOAD_DIR="$SCRATCH/ckpt_offload_${SLURM_JOB_ID}"
mkdir -p $JOB_OFFLOAD_DIR

# räume beim Script-Exit auf, egal ob Erfolg oder Fehler
trap 'echo "Cleanup: rm -rf $JOB_OFFLOAD_DIR"; rm -rf "$JOB_OFFLOAD_DIR"' EXIT

# -------- run lm-evaluation-harness --------
# accelerate launch --multi_gpu \
#   -m lm_eval \
#     --model hf \
#     --model_args "pretrained=$,dtype=bfloat16" \
#     --tasks wikitext \
#     --batch_size auto

    
# lm_eval \
#   --model hf \
#   --model_args "pretrained=$QWEN_QWQ,dtype=bfloat16,device_map=auto" \
#   --tasks wikitext \
#   --batch_size auto 

# # # now run the LLMCBench PPL evaluator
# accelerate launch $CODE_DIR/evaluate_ppl.py \
#     --path       $LLAMA_2_7B \
#     --seqlen      4096


# Run WIKI evaluator
srun python -u $CODE_DIR/evaluate_wiki2.py \
     --path  $Qwen3_30B_A3B
     --batch_size auto 
     #--single_gpu      

# srun python -u  $CODE_DIR/pruning/wanda/main.py \
#   --model "$LLAMA_2_7B" \
#   --prune_method wanda \
#   --sparsity_ratio 0.5 \
#   --sparsity_type unstructured \
#   --save $WORK/models/pruned-models/llama2-7b-wanda50

# -------- sanity check --------
echo "Using Python: $(which python)"
python - <<'PY'
import torch, transformers
print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
PY

# -------- run lm-evaluation-harness --------
# lm_eval \
#   --model hf \
#   --model_args "pretrained=$GEMMA_3_1B,dtype=float16,device_map=auto" \
#   --tasks mmlu \
#   --batch_size auto \

# Run WIKI evaluator
srun python -u $CODE_DIR/evaluate_wiki2.py \
     --path  $QWEN_3_14B_BASE \
#    --single_gpu

# python evaluate_wiki2_gemma.py \
#   --model_path /mnt/lustre/work/geiger/gwb345/models/gemma-3/gemma-3-4b-it \
#   --max_len 2048 \
#   --stride 256 \
#   --torch_dtype torch.bfloat16