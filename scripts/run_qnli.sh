#!/bin/bash
export PYTHONPATH='../LLM-Pruner'
# python evaluate_llama.py -m llama2_7b_hf_llmp --path ../llama/llama-2-7b-llmpruner/pytorch_model.bin
# CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_qnli.py -m llama2_7b_hf_llmp --data_dir data/QNLI --path ../LLM-Pruner/prune_log/llama_prune/pytorch_model.bin
# ../minillm/results/llama/train/minillm/bs8-lr5e-06-G2-N4-NN1-lm1-len512-mp4/pe4_rs0.5_nr256_ln_sr_tm0.2/5000
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
falcon_7b='/mnt/disk1/wjy/falcon_7b'

CUDA_VISIBLE_DEVICES=2 python evaluate_qnli.py -m qwen \
                            --path ../Qwen1.5-MoE-A2.7B \
                            --data_dir data/QNLI \
                            --seqlen 4096
# CUDA_VISIBLE_DEVICES=3 python evaluate_qnli.py -m vicuna_7b_hf_smq --data_dir data/QNLI --path ../smoothquant/results/vicuna