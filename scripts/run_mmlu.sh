#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python evaluate_mmlu.py --path $model_path --data_dir data/MMLU --ntrain 0