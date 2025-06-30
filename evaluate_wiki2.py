#!/usr/bin/env python3
"""
Compute sliding-window perplexity on WikiText-2-raw-v1 (test set).

Key upgrades
------------
* Genau **ein BOS**-Token am Textanfang (modellunabhängig)
* Automatisches Hinzufügen eines Pad-Tokens + Embedding-Resize, falls nötig
* Verwendet modell-spezifische Kontextlänge (model_configs.py) → übersteuerbar via --max_len
* TF32-Matmul bei Ampere+ GPUs, wenn das Modell in BF16 läuft (≈ 10–15 % schneller)
* Robuster OOM-Fallback + exponentielle Batch-Größen-Suche
* Dynamisches Padding nur auf die längste Sequenz im Batch (≠ max_len)

Erwartete PPL-Richtwerte (WikiText-2 test):
    LLaMA-3-8B FP16  … ≈ 5.5
    LLaMA-2-70B 8-bit… ≈ 6-7
    Gemma-3-27B 8-bit… ≈ 5.8
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import time
from typing import List, Tuple

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration, AutoConfig

from model_configs import get_model_cfg


# --------------------------------------------------------------------------- #
# helper functions
# --------------------------------------------------------------------------- #

def try_batch(
    model: torch.nn.Module,
    enc: torch.Tensor,
    windows: List[Tuple[int, int]],
    batch_size: int,
    pad_id: int,
    max_len: int,
) -> bool:
    """Probe-Batch, um die maximale Batch-Größe ohne OOM abzuschätzen."""
    first_device = next(iter(model.hf_device_map.values()))
    sorted_win = sorted(windows, key=lambda w: w[1] - w[0], reverse=True)
    probe = sorted_win[:batch_size]

    input_seqs, target_seqs = [], []
    for begin, end in probe:
        seq = enc[begin:end]
        if seq.size(0) < max_len:
            pad_len = max_len - seq.size(0)
            pad_chunk = torch.full((pad_len,), pad_id, dtype=torch.long)
            seq = torch.cat([pad_chunk, seq])

        inp = seq.clone()
        tgt = seq.clone()
        tgt[:-(end - begin)] = -100
        input_seqs.append(inp)
        target_seqs.append(tgt)

    inp_batch = pad_sequence(input_seqs, batch_first=True, padding_value=pad_id).to(first_device)
    tgt_batch = pad_sequence(target_seqs, batch_first=True, padding_value=-100).to(first_device)

    try:
        with torch.no_grad():
            _ = model(inp_batch, labels=tgt_batch, use_cache=False)
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        raise


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="HF repo oder lokaler Checkpoint")
    ap.add_argument("--max_len", type=int, help="Kontextlänge (default: modellabhängig)")
    ap.add_argument("--stride", type=int, default=256, help="Sliding-Window-Schritt")
    ap.add_argument("--batch_size", default="auto", help='Batch-Größe oder "auto" für autotune')
    ap.add_argument("--single_gpu", action="store_true", help="Alles auf eine GPU statt device_map=auto")
    args = ap.parse_args()

    model_id = os.path.basename(args.path.rstrip("/"))
    cfg = get_model_cfg(model_id)

    start_time = time.time()
    # ---------- tokenizer ----------
    is_gemma = bool(re.match(r"^Gemma-3-", model_id, flags=re.IGNORECASE))
    tokenizer = AutoTokenizer.from_pretrained(
        args.path,
        use_fast=True,
        trust_remote_code=True,
        add_bos_token=False
    )
    added = 0
    if tokenizer.pad_token_id is None:
        added = tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # ---------- model ----------
    quant_cfg = cfg.get("bnb_config")
    torch_dtype = None if cfg.get("quantize") else cfg["torch_dtype"]

    model_cls = Gemma3ForConditionalGeneration if is_gemma else AutoModelForCausalLM

    if args.single_gpu:
        device_map = None
        to_device = "cuda:0"
        max_memory = None
    else:
        device_map = "auto"
        to_device = None
        max_memory = {i: "40GB" for i in range(torch.cuda.device_count())}

    model = model_cls.from_pretrained(
        args.path,
        trust_remote_code=True,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
    )
    if to_device:
        model = model.to(to_device)
    model.eval()

    if added:
        model.resize_token_embeddings(len(tokenizer))

    if (
        model.dtype == torch.bfloat16
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
    ):
        torch.backends.cuda.matmul.allow_tf32 = True


    ########## Debugging ##########
    print(f"Loaded model: {model.config._name_or_path}")
    print("Own Config:")
    print(json.dumps(cfg, indent=2, default=lambda o: repr(o)))
    
    print("Model config:")
    print(f"Tokenizer: {tokenizer.__class__.__name__} (pad_token_id={tokenizer.pad_token_id})")
    print(f"Model dtype: {model.dtype}")
    

    ################################

    first_device = to_device if args.single_gpu else next(iter(model.hf_device_map.values()))

    # ---------- Daten ----------
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer(
        "\n\n".join(ds["text"]),
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids[0]

    if tokenizer.bos_token_id is not None:
        enc = torch.cat([torch.tensor([tokenizer.bos_token_id]), enc])

    total_tokens = enc.size(0)
    max_len = args.max_len or cfg["max_len"]

    windows: List[Tuple[int, int]] = []
    for i in range(0, total_tokens, args.stride):
        end = min(i + args.stride, total_tokens)
        begin = max(0, end - max_len)
        windows.append((begin, end))

    pad_id = tokenizer.pad_token_id

    if args.batch_size == "auto":
        print("Auto-tuning batch_size …")
        best_bs, bs = 1, 1
        while bs <= 32:
            if try_batch(model, enc, windows, bs, pad_id, max_len):
                best_bs = bs
                bs *= 2
            else:
                bs //= 2
                break
        batch_size = best_bs
    else:
        batch_size = int(args.batch_size)
    print(f"Using batch_size = {batch_size}")

    nll = 0.0
    tok_cnt = 0
    idx = 0
    while idx < len(windows):
        batch = windows[idx: idx + batch_size]
        input_seqs, target_seqs, token_counts = [], [], []
        for begin, end in batch:
            seq = enc[begin:end]
            inp = seq.clone()
            tgt = seq.clone()
            new_tokens = end - begin
            tgt[: -new_tokens] = -100
            input_seqs.append(inp)
            target_seqs.append(tgt)
            token_counts.append(new_tokens)

        inp_batch = pad_sequence(
            input_seqs, batch_first=True, padding_value=pad_id
        ).to(first_device)
        tgt_batch = pad_sequence(
            target_seqs, batch_first=True, padding_value=-100
        ).to(first_device)

        try:
            with torch.no_grad():
                outputs = model(
                    inp_batch,
                    labels=tgt_batch,
                    use_cache=False,
                )
                batch_loss = outputs.loss.item()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                print(f"OOM – reducing batch_size to {batch_size}")
                continue
            raise

        nll += batch_loss * sum(token_counts)
        tok_cnt += sum(token_counts)
        idx += batch_size

    ppl = math.exp(nll / tok_cnt)
    print(f"WikiText-2 perplexity: {ppl:.2f}")

    elapsed = time.time() - start_time
    print(f"⏱️  Runtime: {elapsed:.2f} s ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()
