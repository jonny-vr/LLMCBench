#!/usr/bin/env python3
"""
Compute sliding-window perplexity on WikiText-2-raw-v1 (test set) using
8-bit quantization + FP32 CPU offload + device_map="auto" to avoid OOM on large models.
This batched version will be much faster by sending multiple windows per forward pass.
Expected result: ~5.5 for Llama-3-8B (FP16), ~6–7 for Llama-2-70B in 8-bit.
"""

import argparse
import math
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from torch.nn.utils.rnn import pad_sequence


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True,
                   help="HF repo or local checkpoint directory")
    p.add_argument("--max_len", type=int, default=2048,
                   help="context window length")
    p.add_argument("--stride",  type=int, default=256,
                   help="sliding-window stride (new tokens per step)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="how many windows per batch")
    p.add_argument("--torch_dtype", default="torch.float16",
                   help="dtype for model weights, e.g. torch.float16; set to 'torch.float32' to disable")
    p.add_argument("--single_gpu",
                   action="store_true",
                   help="if set, load the model on one GPU instead of auto-sharding")
    args = p.parse_args()

    # ---------- model & tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(
        args.path, use_fast=True, trust_remote_code=True
    )
    # ensure we have a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    torch_dtype = getattr(torch, args.torch_dtype.split('.')[-1])

    # quant_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_enable_fp32_cpu_offload=True,
    # )
    if args.single_gpu:
        # load entire model on GPU:0
        model = AutoModelForCausalLM.from_pretrained(
            args.path,
            device_map=None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to("cuda:0")
    else:
        # shard across all GPUs/CPU automatically
        model = AutoModelForCausalLM.from_pretrained(
            args.path,
            # quantization_config=quant_config, # nur für größere modelle
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    # max_memory={0: "40GB",    # map visible GPU 0 → 40 GB
    #             1: "40GB",    # map visible GPU 1 → 40 GB
    #             "cpu": "120GB"},  # spill the rest to CPU
    # <<< INSERT HERE >>>
    # Print out which model we just loaded
    print(f"Loaded model: {model.config._name_or_path}")
    model.eval()

    # ---------- data & tokenization ----------
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # join texts but we'll slice on the token level below
    enc = tokenizer(
        "\n\n".join(ds["text"]),
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids[0]

    total_tokens = enc.size(0)

    # build all sliding-window (begin, end) index pairs
    windows = []
    for i in range(0, total_tokens, args.stride):
        end = min(i + args.stride, total_tokens)
        begin = max(0, end - args.max_len)
        windows.append((begin, end))

    # ---------- batched evaluation ----------
    nll, tok_cnt = 0.0, 0
    pad_id = tokenizer.pad_token_id

    for idx in range(0, len(windows), args.batch_size):
        batch = windows[idx: idx + args.batch_size]
        input_seqs, target_seqs, token_counts = [], [], []

        for (begin, end) in batch:
            seq = enc[begin:end]
            inp = seq.clone()
            tgt = seq.clone()
            new_tokens = end - begin  # always ≤ stride
            # mask out all but the new tokens in this window
            tgt[: -new_tokens] = -100
            input_seqs.append(inp)
            target_seqs.append(tgt)
            token_counts.append(new_tokens)

        # pad to the longest sequence in this batch
        inp_batch = pad_sequence(
            input_seqs, batch_first=True, padding_value=pad_id)
        tgt_batch = pad_sequence(
            target_seqs, batch_first=True, padding_value=-100)

        inp_batch = inp_batch.to(model.device)
        tgt_batch = tgt_batch.to(model.device)

        with torch.no_grad():
            output = model(inp_batch, labels=tgt_batch)
            # output.loss is averaged over all non-masked tokens
            batch_loss = output.loss.item()

        # scale back up to token-level NLL
        batch_token_sum = sum(token_counts)
        nll += batch_loss * batch_token_sum
        tok_cnt += batch_token_sum

    ppl = math.exp(nll / tok_cnt)
    print(f"WikiText-2 perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
