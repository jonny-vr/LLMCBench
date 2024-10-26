#!/bin/bash
export PYTHONPATH='../LLM-Pruner'
# CUDA_VISIBLE_DEVICES=1,2,4 python evaluate_mnli.py -m vicuna_7b_llmp  --data_dir data/MNLI --path ../LLM-Pruner/prune_log/vicuna_prune/pytorch_model.bin
llama_7b='/mnt/disk1/hg/huggingface/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348'
llama_13b='/mnt/disk1/hg/huggingface/cache/models--decapoda-research--llama-13b-hf/snapshots/438770a656712a5072229b62256521845d4de5ce'
llama_30b='/mnt/disk1/hg/huggingface/cache/models--decapoda-research--llama-30b-hf/snapshots/f991780f9362b2fcaefad066cd235058844562b7'
opt_1b3='/mnt/disk1/yg/opt/opt-1.3b'
opt_2b7='/mnt/disk1/yg/opt/opt-2.7b'
opt_6b7='/mnt/disk1/hg/huggingface/cache/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0'
opt_13b='/mnt/disk1/hg/huggingface/cache/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5'
opt_30b='/mnt/disk1/hg/huggingface/cache/models--facebook--opt-30b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546'
llama2_13b='/mnt/disk1/wjy/Llama-2-13b-hf'
vicuna_13b='/mnt/disk1/yg/vicuna-13b-v1.5'

CUDA_VISIBLE_DEVICES=1 python evaluate_mnli.py \
                    --path '/mnt/disk1/yg/llama/llama-2-7b-hf' \
                    --seqlen 4096 --data_dir data/MNLI