#!/usr/bin/env python3
import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


def format_example(df, idx, include_answer=True):
    """
    Build a single example prompt from QNLI dataframe.
    Uses columns: question, sentence (premise), label (entailment/not_entailment).
    """
    prompt = "Question: " + df.iloc[idx, 1]
    prompt += "\nSentence: " + df.iloc[idx, 2]
    prompt += "\nAnswer:"
    if include_answer:
        label = df.iloc[idx, 3]
        # map QNLI labels to text
        ans = "A. yes" if label == "entailment" else "B. no"
        prompt += f" {ans}\n\n"
    return prompt


def gen_prompt(train_df, k):
    """
    Create k-shot demonstration prompt from first k rows of train_df.
    """
    base = "Please identify whether the sentence entails the question."
    base += " The answer should be exactly 'A. yes' or 'B. no'.\n\n"
    examples = []
    for i in range(k):
        examples.append(format_example(train_df, i, include_answer=True))
    return base + "".join(examples)


def eval_qnli(args, model, tokenizer, train_df, test_df):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    correct = []
    for i in tqdm(range(len(test_df)), desc="Evaluating QNLI"):
        # build prompt
        k = args.ntrain
        demo = gen_prompt(train_df, k)
        test_prompt = format_example(test_df, i, include_answer=False)
        prompt = demo + test_prompt

        # tokenize and possibly trim demos if too long
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"].to(device)
        while input_ids.shape[-1] > args.seqlen and k > 0:
            k -= 1
            demo = gen_prompt(train_df, k)
            prompt = demo + test_prompt
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
                "input_ids"].to(device)

        # get logits of last token
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :].flatten()

        # probability for 'A' and 'B'
        id_A = tokenizer("A").input_ids[-1]
        id_B = tokenizer("B").input_ids[-1]
        probs = torch.softmax(torch.tensor(
            [logits[id_A], logits[id_B]]), dim=0).cpu().numpy()
        pred = np.argmax(probs)

        gold = test_df.iloc[i, 3]
        gold_idx = 0 if gold == "entailment" else 1
        correct.append(pred == gold_idx)

    accuracy = np.mean(correct)
    print(f"QNLI (ntrain={args.ntrain}) accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot/few-shot QNLI evaluation with causal LMs"
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Path to local model checkpoint or HF repo"
    )
    parser.add_argument(
        "--data_dir", "-d", type=str, default="data/QNLI",
        help="Directory with QNLI tsv files (train.tsv, dev.tsv)"
    )
    parser.add_argument(
        "--ntrain", "-k", type=int, default=0,
        help="Number of in-context examples (shots)"
    )
    parser.add_argument(
        "--seqlen", type=int, default=2048,
        help="Maximum total sequence length"
    )
    parser.add_argument(
        "--torch_dtype", type=str, default="torch.float16",
        help="Dtype for model weights: torch.float16 or torch.float32"
    )
    args = parser.parse_args()

    # load tokenizer and model
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
    ).to("cuda").eval()

    # load dataframes
    train_df = pd.read_csv(
        os.path.join(args.data_dir, "train.tsv"),
        sep="\t",
        on_bad_lines="skip"
    )[: args.ntrain]
    test_df = pd.read_csv(
        os.path.join(args.data_dir, "dev.tsv"),
        sep="\t",
        on_bad_lines="skip"
    )

    eval_qnli(args, model, tokenizer, train_df, test_df)


if __name__ == "__main__":
    main()
