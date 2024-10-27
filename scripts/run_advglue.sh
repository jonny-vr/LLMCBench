#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python evaluate_advglue.py --path  $model_path --ntrain 0
