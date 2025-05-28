#!/usr/bin/env python3
import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_piqa_df(split: str):
    """
    Load PIQA from Hugging Face and return a pandas DataFrame with columns:
    - goal (str)
    - sol1 (str)
    - sol2 (str)
    - label (int, 0 or 1)
    """
    ds = load_dataset("piqa", split=split, trust_remote_code=True)
    df = ds.to_pandas()[["goal", "sol1", "sol2", "label"]]
    return df


def score_option(model, tokenizer, context: str, completion: str, device: str) -> float:
    """
    Compute total log-prob of `completion` given `context`.
    """
    # Concatenate context and completion
    text = context + " " + completion
    encoded = tokenizer(text, return_tensors="pt",
                        add_special_tokens=False).to(device)
    input_ids = encoded.input_ids

    # Get model logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Shift tokens and labels
    shift_logits = logits[:, :-1, :].squeeze(0)
    shift_labels = input_ids[:, 1:].squeeze(0)

    # Compute log-probs
    logprobs = torch.log_softmax(shift_logits, dim=-1)

    # Identify completion token IDs
    comp_ids = tokenizer(completion, add_special_tokens=False).input_ids
    k = len(comp_ids)
    seq_len = input_ids.size(1)

    # Completion tokens occupy the last k positions of shift_labels
    start = seq_len - 1 - k
    total_logprob = 0.0
    for i, token_id in enumerate(comp_ids, start=start):
        total_logprob += logprobs[i, token_id].item()

    return total_logprob


def main(args):
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model & tokenizer
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

    # Load PIQA data
    split = "validation" if args.split == "val" else "test"
    df = load_piqa_df(split)

    # Evaluate zero-shot
    correct = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating PIQA"):
        goal = row["goal"]
        options = [row["sol1"], row["sol2"]]
        scores = [score_option(model, tokenizer, goal, opt, device)
                  for opt in options]
        pred = int(torch.tensor(scores).argmax())
        if pred == int(row["label"]):
            correct += 1

    accuracy = correct / len(df)
    print(f"PIQA ({args.split}) accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of causal LMs on PIQA"
    )
    parser.add_argument(
        "--path", required=True,
        help="Path to model checkpoint or HuggingFace repo"
    )
    parser.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Which split to evaluate: 'val' (validation) or 'test'"
    )
    parser.add_argument(
        "--torch_dtype", default="torch.float16",
        help="torch dtype, e.g. torch.float16 or torch.float32"
    )
    args = parser.parse_args()
    main(args)

