#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python evaluate_mnli.py --path $model_path -data_dir data/MNLI