#!/usr/bin/env python3
import argparse
import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def load_hellaswag_df(split: str):
    ds = load_dataset(
        "hellaswag",
        split=split,
        trust_remote_code=True,
    )
    df = ds.to_pandas()
    # Expand endings list into opt0–opt3
    endings_df = pd.DataFrame(df["endings"].tolist(),
                              columns=[f"opt{i}" for i in range(4)])
    return pd.concat([df[["ctx", "label"]].reset_index(drop=True),
                      endings_df.reset_index(drop=True)], axis=1)

def score_ending(model, tokenizer, ctx, ending, device):
    # Tokenize context + ending together
    encoded = tokenizer(ctx + ending,
                        return_tensors="pt",
                        add_special_tokens=False).to(device)
    input_ids = encoded.input_ids
    # We want log-probs for the ending tokens only
    # So run model with labels=input_ids => loss averages over all tokens;
    # Instead compute logits & gather logprobs manually
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)
    # shift so that logits[t] predicts token t
    shift_logits = logits[:, :-1, :].squeeze(0)         # (seq_len-1, vocab)
    shift_labels = input_ids[:, 1:].squeeze(0)           # (seq_len-1)
    # full sequence length
    seq_len = input_ids.size(1)
    # compute logprobs for all tokens
    logprobs = torch.log_softmax(shift_logits, dim=-1)   # (seq_len-1, vocab)
    # identify which positions correspond to the ending
    # we assume `ending` tokens are the last `k` tokens
    # find k by tokenizing ending alone
    ending_ids = tokenizer(ending, add_special_tokens=False).input_ids
    k = len(ending_ids)
    # ending positions are last k positions of shift_labels
    ending_positions = list(range(seq_len-1-k, seq_len-1))
    # sum logprobs at those positions for the correct token
    total = 0.0
    for pos, token_id in zip(ending_positions, ending_ids):
        total += logprobs[pos, token_id].item()
    return total

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.path,
        use_fast=False,
        add_bos_token=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        torch_dtype=getattr(torch, args.torch_dtype.split('.')[-1]),
        trust_remote_code=True,
    ).to(device).eval()

    # Load HellaSwag
    split = "validation" if args.split == "val" else "test"
    df = load_hellaswag_df(split)

    # Evaluate
    correct = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ctx = row["ctx"]
        scores = []
        for i in range(4):
            ending = row[f"opt{i}"]
            scores.append(score_ending(model, tokenizer, ctx, ending, device))
        pred = int(torch.tensor(scores).argmax())
        if pred == int(row["label"]):
            correct += 1

    acc = correct / len(df)
    print(f"HellaSwag ({args.split}) accuracy: {acc:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True,
                   help="model checkpoint path")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="which split to evaluate (val=validation)")
    p.add_argument("--torch_dtype", default="torch.float16",
                   help="torch dtype, e.g. torch.float16 or torch.float32")
    args = p.parse_args()
    main(args)

