#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python evaluate_qnli.py --path $model_path --data_dir data/QNLI