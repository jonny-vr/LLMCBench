#!/bin/bash
#SBATCH -J distill_llama2_3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # ein Task (Prozess) pro GPU
#SBATCH --cpus-per-task=16             # pro Prozess
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1                  # vier GPUs
#SBATCH --mem=80G                    # Gesamt-RAM (optional anpassen)
#SBATCH --time=0-12:00:00             # z.B. 2 Tage
#SBATCH --output=logs/distill.%j.out
#SBATCH --error=logs/distill.%j.err


# Fail-Fast / Debug
echo "Starte Job $SLURM_JOB_ID am $(date)"

# -------- Conda & HF setup --------
export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate /home/geiger/gwb082/.conda/envs/thesis

export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
CODE_DIR="/home/geiger/gwb082/Jonathans_Thesis/LLMCBench"
MODEL_DIR="/home/geiger/gwb082/LLMs"

#   "$MODEL_DIR/llama-2/llama-2-7b-hf"
#   "$MODEL_DIR/llama-2/llama-2-13b-hf"
#   "$MODEL_DIR/gemma-3/gemma-3-1b-pt"
#   "$MODEL_DIR/gemma-3/gemma-3-4b-pt"
#   "$MODEL_DIR/gemma-3/gemma-3-12b-pt"
# "$MODEL_DIR/llama-3/llama-3.1-8b-hf" 
#   "$MODEL_DIR/Qwen/Qwen3-0.6B-Base"
#   "$MODEL_DIR/Qwen/Qwen3-1.7B-Base"
#   "$MODEL_DIR/Qwen/Qwen3-4B-Base"
#   "$MODEL_DIR/Qwen/Qwen3-8B-Base"
#   "$MODEL_DIR/Qwen/Qwen3-14B-Base"
# "$MODEL_DIR/deepseek/DeepSeek-R1-Distill-Llama-8B"
# "$MODEL_DIR/deepseek/deepseek-moe-16b-base"
# -------- Liste aller Modelle --------
MODELS=(
  "$MODEL_DIR/llama-2/llama-2-70b-hf"
  "$MODEL_DIR/llama-3/Llama-3.1-70B"
  "$MODEL_DIR/gemma-3/gemma-3-27b-pt"
  "$MODEL_DIR/llama-4/Llama-4-Scout-17B-16E"
  "$MODEL_DIR/Qwen/Qwen3-32B"
  "$MODEL_DIR/Qwen/Qwen-QwQ-32B"
  "$MODEL_DIR/Qwen/Qwen3-30B-A3B-Base"
  "$MODEL_DIR/deepseek/DeepSeek-R1-Distill-Llama-70B"
)


LLAMA_2_7B="$MODEL_DIR/llama-2/llama-2-7b-hf"
LLAMA_2_13B="$MODEL_DIR/llama-2/llama-2-13b-hf"
LLAMA_2_70B="$MODEL_DIR/llama-2/llama-2-70b-hf"
LLAMA_3_8B="$MODEL_DIR/llama-3/llama-3.1-8b-hf"
LLAMA_3_70B="$MODEL_DIR/llama-3/Llama-3.1-70B"
LLAMA_4_SCOUT_109B_17B="$MODEL_DIR/llama-4/Llama-4-Scout-17B-16E"
QWEN_3_0_6B_BASE="$MODEL_DIR/Qwen/Qwen3-0.6B-Base"
QWEN_3_1_7B_BASE="$MODEL_DIR/Qwen/Qwen3-1.7B-Base"
QWEN_3_4B_BASE="$MODEL_DIR/Qwen/Qwen3-4B-Base"
QWEN_3_8B_BASE="$MODEL_DIR/Qwen/Qwen3-8B-Base"
QWEN_3_14B_BASE="$MODEL_DIR/Qwen/Qwen3-14B-Base"
QWEN_3_32B="$MODEL_DIR/Qwen/Qwen3-32B"
QWEN_3_0_6B_IT="$MODEL_DIR/Qwen/Qwen3-0.6B"
QWEN_3_1_7B_IT="$MODEL_DIR/Qwen/Qwen3-1.7B"
QWEN_3_4B_IT="$MODEL_DIR/Qwen/Qwen3-4B"
QWEN_3_8B_IT="$MODEL_DIR/Qwen/Qwen3-8B"
QWEN_3_14B_IT="$MODEL_DIR/Qwen/Qwen3-14B"
QWEN_QWQ="$MODEL_DIR/Qwen/Qwen-QwQ-32B"
Qwen3_30B_A3B="$MODEL_DIR/Qwen/Qwen3-30B-A3B-Base"
DEEPSEEK_MOE_16B_BASE="$MODEL_DIR/deepseek/deepseek-moe-16b-base"
DEEPSEEK_R1_LLAMA_3_8B="$MODEL_DIR/deepseek/DeepSeek-R1-Distill-Llama-8B"
DEEPSEEK_R1_LLAMA_3_70B="$MODEL_DIR/deepseek/DeepSeek-R1-Distill-Llama-70B"

CHECKPOINT="/home/geiger/gwb082/Jonathans_Thesis/LLMCBench/finetuned_wiki2_seq1024"

# srun python $CODE_DIR/finetune_wiki2.py

# # Run WIKI evaluator
# srun python -u $CODE_DIR/evaluate_wiki2.py \
#      --path  $CHECKPOINT \
#      --batch_size auto 
#      #--single_gpu

# ─── Edit only these three ────────────────────────────────────────────
LR=1e-4
ALPHA=0.5
WIKI_PCT=2
SEQ_LEN=1024
LAYER_RATIO=0.27
# ─────────────────────────────────────────────────────────────────────

# Der Rest passt sich automatisch an
# RUN_NAME="distill_lr${LR}_wiki${WIKI_PCT}_alpha${ALPHA}_seq${SEQ_LEN}_layer${LAYER_RATIO}"
# OUTPUT_DIR="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/distilled/llama2_3.5b_${RUN_NAME}"

# echo "=== Starte Distillation von LLaMA-2 7B ==="
# echo "  Teacher:   /home/geiger/gwb082/LLMs/llama-2/llama-2-7b-hf"
# echo "  Output:    ${OUTPUT_DIR}"
# echo "  Run name:  ${RUN_NAME}"
# echo "=================================================="
# echo

# srun torchrun \
#   --nnodes=1 \
#   --nproc_per_node=1 \
#   --master_port=29510 \
#   /home/geiger/gwb082/Jonathans_Thesis/LLMCBench/distillation/distill_llama_student.py \
#     --teacher_path        /home/geiger/gwb082/LLMs/llama-2/llama-2-7b-hf \
#     --output_dir         "${OUTPUT_DIR}" \
#     --dataset_name       wikipedia \
#     --dataset_config     20220301.en \
#     --warmup_steps       500 \
#     --wiki_pct           "${WIKI_PCT}" \
#     --layer_ratio        "${LAYER_RATIO}" \
#     --max_seq_length     "${SEQ_LEN}" \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate      "${LR}" \
#     --num_train_epochs   3 \
#     --temperature        2.0 \
#     --alpha_distill      "${ALPHA}" \
#     --wandb_project      llama-distillation \
#     --wandb_entity       jonathan-von-rad \
#     --wandb_run_name     "${RUN_NAME}"




# -------- Schleife über alle Modelle --------
# for MODEL_PATH in "${MODELS[@]}"; do
#     MODEL_NAME=$(basename "$MODEL_PATH")
#     echo "====> Evaluiere $MODEL_NAME"

#     OUT_LOG="logs/${MODEL_NAME}.${SLURM_JOB_ID}.out"
#     ERR_LOG="logs/${MODEL_NAME}.${SLURM_JOB_ID}.err"

#     python -u "$CODE_DIR/evaluate_wiki2.py" \
#         --path "$MODEL_PATH" \
#         --batch_size auto \
#         >"$OUT_LOG" 2>"$ERR_LOG"

#     echo "    → Fertig $MODEL_NAME: $(tail -n1 "$OUT_LOG")"
# done

# echo "Alle Modelle abgearbeitet um $(date)"


# srun python -u  $CODE_DIR/pruning/wanda/main.py \
#   --model "$GEMMA_3_4B" \
#   --prune_method wanda \
#   --sparsity_ratio 0.5 \
#   --sparsity_type unstructured \
#   --save /home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Gemma-3-4B-Wanda-0.5 \

# -------- run lm-evaluation-harness --------
lm_eval \
  --model hf \
  --model_args "pretrained=$QWEN_3_4B_BASE,dtype=bfloat16,device_map=auto" \
  --tasks gsm8k \
  --batch_size auto 
