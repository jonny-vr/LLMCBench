#!/usr/bin/env python
import argparse
import os
import time
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoTokenizer, AutoModelForCausalLM

choices = ["A", "B", "C", "D"]


def format_subject(subject: str) -> str:
    return " " + " ".join(subject.split("_"))


def format_example(df: pd.DataFrame, idx: int, include_answer: bool = True) -> str:
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += f"\n{choices[j]}. {df.iloc[idx, j + 1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k + 1]}\n\n"
    return prompt


def gen_prompt(train_df: pd.DataFrame, subject: str, k: int = -1) -> str:
    k = train_df.shape[0] if k == -1 else k
    header = (
        f"The following are multiple choice questions (with answers) about{format_subject(subject)}.\n\n"
    )
    return header + "".join(format_example(train_df, i) for i in range(k))


@torch.inference_mode()
def eval_subject_batched(
    args, subject: str, model, tokenizer, dev_df: pd.DataFrame,
    test_df: pd.DataFrame, choice_ids: list[int], batch_size: int = 8
):
    cors, all_probs = [], []
    # Precompute few-shot context once per subject
    shot_prompts = [format_example(dev_df, i) for i in range(args.ntrain)]
    few_shot = (
        f"The following are multiple choice questions (with answers) about{format_subject(subject)}.\n\n"
        + "".join(shot_prompts)
    )

    # Build tails and labels
    tails, labels = [], []
    for i in range(len(test_df)):
        tail = format_example(test_df, i, include_answer=False)
        tails.append(few_shot + tail)
        labels.append(test_df.iloc[i, -1])

    # Batch processing
    for i in range(0, len(tails), batch_size):
        batch_prompts = tails[i: i + batch_size]
        batch_labels = labels[i: i + batch_size]

        # Tokenize and move to GPU
        tok = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=args.seqlen,
        ).to("cuda")

        # Forward pass
        outputs = model(**tok)
        logits = outputs.logits  # shape (B, L, V)

        # Extract last token logits per example
        seq_lens = tok.attention_mask.sum(dim=1) - 1  # (B,)
        last_logits = logits[torch.arange(len(seq_lens)), seq_lens]  # (B, V)

        # Compute choice probabilities
        probs = torch.softmax(
            last_logits[:, choice_ids].float(), dim=1
        ).cpu().numpy()  # (B, 4)

        # Collect results
        for p_vec, lbl in zip(probs, batch_labels):
            pred = choices[int(np.argmax(p_vec))]
            cors.append(pred == lbl)
            all_probs.append(p_vec)

    acc = float(np.mean(cors))
    print(f"Average accuracy {acc:.3f} – {subject}", flush=True)
    return np.array(cors), acc, np.array(all_probs)


def main(args):
    start_all = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        args.path,
        use_fast=False,
        add_bos_token=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        trust_remote_code=True,
    ).to("cuda").eval()


    print(f"Loaded model dtype: {next(model.parameters()).dtype}", flush=True)

    # Cache choice token IDs
    choice_ids = [tokenizer(c).input_ids[-1] for c in choices]

    subjects = sorted(
        f.split("_test.csv")[0]
        for f in os.listdir(os.path.join(args.data_dir, "test"))
        if f.endswith("_test.csv")
    )

    all_cors = []
    subcat_cors = {s: [] for subs in subcategories.values() for s in subs}
    cat_cors = {c: [] for c in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", f"{subject}_dev.csv"),
            header=None,
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", f"{subject}_test.csv"),
            header=None,
        )

        cors, acc, probs = eval_subject_batched(
            args, subject, model, tokenizer, dev_df, test_df,
            choice_ids, batch_size=args.batch_size
        )

        all_cors.append(cors)
        for sc in subcategories[subject]:
            subcat_cors[sc].append(cors)
            for cat, subs in categories.items():
                if sc in subs:
                    cat_cors[cat].append(cors)

    for sc, lists in subcat_cors.items():
        print(
            f"Average accuracy {np.mean(np.concatenate(lists)):.4f} – {sc}", flush=True)
    for cat, lists in cat_cors.items():
        print(
            f"Average accuracy {np.mean(np.concatenate(lists)):.4f} – {cat}", flush=True)

    total_acc = np.mean(np.concatenate(all_cors))
    print(f"Overall MMLU accuracy: {total_acc:.4f}", flush=True)
    print(
        f"Total evaluation time: {time.perf_counter() - start_all:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int,
                        default=0, help="few-shot examples")
    parser.add_argument("--data_dir", "-d", required=True, help="dataset root")
    parser.add_argument("--path", required=True, help="model checkpoint path")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch_size", type=int,
                        default=8, help="batch size for eval")
    args = parser.parse_args()
    main(args)

