#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python evaluate_tQA.py --path $model_path --input_path data/TruthfulQA/TruthfulQA.csv