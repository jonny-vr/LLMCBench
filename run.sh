#!/bin/bash
#SBATCH -J eval_wiki2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:2
#SBATCH --mem=40G
#SBATCH --time=1-12:00
#SBATCH --output=logs/eval_wiki2.%j.out
#SBATCH --error=logs/eval_wiki2.%j.err

echo "===== DEBUG START ====="
echo "This job’s HOME = $HOME"
echo "Listing cond a.sh under the absolute path:"
ls -ld /home/geiger/gwb345/miniconda/etc/profile.d/conda.sh \
   && echo "→ Found it!" \
   || echo "→ Not found on compute node!"
echo "Listing your $HOME tree a bit:"
ls -l $HOME || echo "Cannot ls $HOME"
echo "===== DEBUG END ====="

# -------- Conda & HF setup --------
export PATH="/home/geiger/gwb345/miniconda/bin:$PATH"
source "/home/geiger/gwb345/miniconda/etc/profile.d/conda.sh"
conda activate thesis

export HF_HUB_TOKEN="hf_XXX"  # Replace with your actual Hugging Face token
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export CODE_DIR="/mnt/lustre/work/geiger/gwb345/LLMCBench"

LLAMA_2_7B="/mnt/lustre/work/geiger/gwb345/models/llama-2/llama-2-7b-hf"
LLAMA_2_13B="/mnt/lustre/work/geiger/gwb345/models/llama-2/llama-2-13b-hf"
LLAMA_2_70B="/mnt/lustre/work/geiger/gwb345/models/llama-2/llama-2-70b-hf"
LLAMA_3_8B="/mnt/lustre/work/geiger/gwb345/models/llama-3/llama-3.1-8b-hf"
LLAMA_3_70B="/mnt/lustre/work/geiger/gwb345/models/llama-3/Llama-3.1-70B"
GEMMA_3_1B="/mnt/lustre/work/geiger/gwb345/models/gemma-3/gemma-3-1b-it"
GEMMA_3_4B="/mnt/lustre/work/geiger/gwb345/models/gemma-3/gemma-3-4b-it"
GEMMA_3_12B="/mnt/lustre/work/geiger/gwb345/models/gemma-3/gemma-3-12b-it"
GEMMA_3_27B="/mnt/lustre/work/geiger/gwb345/models/gemma-3/gemma-3-27b-it"
QWEN_3_0_6B="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-0.6B"
QWEN_3_0_6_BASE="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-0.6B-Base"
QWEN_3_1_7B="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-1.7B"
QWEN_3_1_7B_BASE="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-1.7B-Base"
QWEN_3_4B="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-4B"
QWEN_3_4B_BASE="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-4B-Base"
QWEN_3_8B="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-8B"
QWEN_3_8B_BASE="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-8B-Base"
QWEN_3_14B="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-14B"
QWEN_3_14B_BASE="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-14B-Base"
QWEN_3_32B="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen3-32B"
QWEN_QWQ="/mnt/lustre/work/geiger/gwb345/models/Qwen/Qwen-QWQ-32B"

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