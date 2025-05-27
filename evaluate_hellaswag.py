#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate_mmlu_batched import eval_subject_batched, choices


def main(args):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.path,
        use_fast=False,
        add_bos_token=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        torch_dtype=args.torch_dtype,
        trust_remote_code=True,
    ).to("cuda").eval()

    # Cache choice token IDs
    choice_ids = [tokenizer(c).input_ids[-1] for c in choices]

    # Load CSV
    csv_path = os.path.join(args.csv_dir, f"hellaswag_{args.split}.csv")
    df = pd.read_csv(csv_path, header=None)

    # Zero-shot: no few-shot context
    dev_df = df.iloc[:0, :].copy()
    test_df = df.copy()

    # Map numeric 0–3 in the last column to letter "A"–"D"
    letter_map = {i: choices[i] for i in range(len(choices))}
    test_df.iloc[:, -1] = test_df.iloc[:, -1].astype(int).map(letter_map)

    # Ensure zero-shot by setting ntrain to 0
    args.ntrain = 0

    # Run batched evaluation
    _, acc, _ = eval_subject_batched(
        args,
        "hellaswag",
        model,
        tokenizer,
        dev_df,
        test_df,
        choice_ids,
        batch_size=args.batch_size
    )
    print(f"HellaSwag ({args.split}) accuracy: {acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="model checkpoint path")
    p.add_argument("--csv_dir", required=True,
                   help="directory containing hellaswag_<split>.csv")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="which split to evaluate (use 'val' for labeled dev set)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--torch_dtype", default="torch.float16",
                   help="torch dtype, e.g. torch.float16 or torch.float32")
    args = p.parse_args()

    # Convert torch_dtype string to actual dtype
    dtype_attr = args.torch_dtype.split('.')[-1]
    args.torch_dtype = getattr(torch, dtype_attr)

    main(args)

