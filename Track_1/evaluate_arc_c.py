#!/usr/bin/env python3
import argparse
import os
import torch
import time

import pandas as pd
import numpy as np
from Track_1.evaluate_mmlu_batched import eval_subject_batched, choices
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.path,
        use_fast=False,
        add_bos_token=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda").eval()

    print(f"Loaded model: {args.path}", flush=True)
    choice_ids = [tokenizer(c).input_ids[-1] for c in choices]
    
    start_time = time.perf_counter() # start timer

    dev_path = os.path.join(args.data_dir, "dev", "arc_challenge_dev.csv")
    test_path = os.path.join(args.data_dir, "test", "arc_challenge_test.csv")
    dev_df = pd.read_csv(dev_path, header=None)[: args.ntrain]
    test_df = pd.read_csv(test_path, header=None)

    cors, acc, _ = eval_subject_batched(
        args, "arc_challenge", model, tokenizer, dev_df, test_df, choice_ids, batch_size=args.batch_size
    )
    print(f"ARC-Challenge accuracy: {acc:.4f}")
    end_time = time.perf_counter()
    print(f"Total evaluation time: {end_time - start_time:.2f} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=0)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)

