#!/bin/bash
#SBATCH -J eval_wiki_all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH --time=0-4:00
#SBATCH --output=logs/eval_wiki_all.%j.out
#SBATCH --error=logs/eval_wiki_all.%j.err

# Fail-Fast / Debug
set -euo pipefail
echo "Starte Job $SLURM_JOB_ID am $(date)"

# -------- Conda & HF setup --------
export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate /home/geiger/gwb082/.conda/envs/thesis

export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
CODE_DIR="/home/geiger/gwb082/Jonathans_Thesis/LLMCBench"
MODEL_DIR="/home/geiger/gwb082/LLMs"

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

# -------- Liste aller Modelle --------
MODELS=(
  "$MODEL_DIR/llama-2/llama-2-7b-hf"
  "$MODEL_DIR/llama-2/llama-2-13b-hf"
  "$MODEL_DIR/llama-2/llama-2-70b-hf"
  "$MODEL_DIR/llama-3/llama-3.1-8b-hf"
  "$MODEL_DIR/llama-3/Llama-3.1-70B"
  "$MODEL_DIR/gemma-3/gemma-3-1b-pt"
  "$MODEL_DIR/gemma-3/gemma-3-4b-pt"
  "$MODEL_DIR/gemma-3/gemma-3-12b-pt"
  "$MODEL_DIR/gemma-3/gemma-3-27b-pt"
  "$MODEL_DIR/llama-4/Llama-4-Scout-17B-16E"
  "$MODEL_DIR/Qwen/Qwen3-0.6B-Base"
  "$MODEL_DIR/Qwen/Qwen3-1.7B-Base"
  "$MODEL_DIR/Qwen/Qwen3-4B-Base"
  "$MODEL_DIR/Qwen/Qwen3-8B-Base"
  "$MODEL_DIR/Qwen/Qwen3-14B-Base"
  "$MODEL_DIR/Qwen/Qwen3-32B"
  "$MODEL_DIR/Qwen/Qwen-QwQ-32B"
  "$MODEL_DIR/Qwen/Qwen3-30B-A3B-Base"
  "$MODEL_DIR/deepseek/deepseek-moe-16b-base"
  "$MODEL_DIR/deepseek/DeepSeek-R1-Distill-Llama-8B"
  "$MODEL_DIR/deepseek/DeepSeek-R1-Distill-Llama-70B"
)



# Logs-Ordner anlegen


# -------- Schleife über alle Modelle --------
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "====> Evaluiere $MODEL_NAME"

    OUT_LOG="logs/${MODEL_NAME}.${SLURM_JOB_ID}.out"
    ERR_LOG="logs/${MODEL_NAME}.${SLURM_JOB_ID}.err"

    python -u "$CODE_DIR/evaluate_wiki2.py" \
        --path "$MODEL_PATH" \
        --batch_size auto \
        >"$OUT_LOG" 2>"$ERR_LOG"

    echo "    → Fertig $MODEL_NAME: $(tail -n1 "$OUT_LOG")"
done

echo "Alle Modelle abgearbeitet um $(date)"


# srun python -u  $CODE_DIR/pruning/wanda/main.py \
#   --model "$LLAMA_2_7B" \
#   --prune_method wanda \
#   --sparsity_ratio 0.5 \
#   --sparsity_type unstructured \
#   --save /home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/llama2-7b-wanda50

# -------- run lm-evaluation-harness --------
# lm_eval \
#   --model hf \
#   --model_args "pretrained=$GEMMA_3_1B,dtype=float16,device_map=auto" \
#   --tasks mmlu \
#   --batch_size auto \
