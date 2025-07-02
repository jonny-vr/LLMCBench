#!/usr/bin/env python3
"""
Compute fixed-block perplexity on WikiText-2-raw-v1 (test set) in one-pass-per-block mode.

Key features:
* Single BOS token at text front (model‑agnostic)
* Automatic pad‑token addition and embedding resize if needed
* Uses model‑specific context length from model_configs.py (overrideable via --max_len)
* Optional batch‑size autotuning (finds largest batch that fits in VRAM)
* Simple next‑token loss over fixed‑length chunks (no sliding windows)
"""

from __future__ import annotations
import argparse, json, math, os, re, time
from typing import Optional, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Gemma3ForConditionalGeneration,
    Llama4ForConditionalGeneration,
)

from model_configs import get_model_cfg

# --------------------------------------------------------------------------- #
# helper: probe whether a given batch size fits                                #
# --------------------------------------------------------------------------- #

def try_batch(
    model: torch.nn.Module,
    ids: torch.Tensor,  # [nsamples, seq_len]
    batch_size: int,
    first_device: torch.device,
    pad_id: int,
) -> bool:
    """Run a single forward with <batch_size> blocks to see if it OOMs."""
    bs = min(batch_size, ids.size(0))
    probe = ids[:bs].to(first_device)
    tgt   = probe.clone()
    tgt[:, :-1] = -100
    attn_mask = torch.ones_like(probe).to(first_device)
    try:
        with torch.no_grad():
            _ = model(
                probe,
                attention_mask=attn_mask,
                labels=tgt,
                use_cache=False,
            )
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        raise

# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="HF repo or local checkpoint")
    ap.add_argument("--batch_size", default="auto", help="Batch size or 'auto' to autotune")
    ap.add_argument("--single_gpu", action="store_true", help="Force single‑GPU loading")
    args = ap.parse_args()

    model_id = os.path.basename(args.path.rstrip("/"))
    cfg = get_model_cfg(model_id)

    start_time = time.time()
    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(
        args.path, use_fast=True, trust_remote_code=True, add_bos_token=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    pad_id = tokenizer.pad_token_id

    # ---------- Model ----------
    quant_cfg  = cfg.get("bnb_config")
    torch_dtype = None if cfg.get("quantize") else cfg["torch_dtype"]

    is_llama4 = bool(re.match(r"^Llama-4-Scout", model_id))
    is_gemma  = bool(re.match(r"^Gemma-3-", model_id, flags=re.IGNORECASE))
    if is_llama4:
        model_cls = Llama4ForConditionalGeneration
    elif is_gemma:
        model_cls = Gemma3ForConditionalGeneration
    else:
        model_cls = AutoModelForCausalLM

    if args.single_gpu:
        device_map, to_device, max_mem = None, "cuda:0", None
    else:
        device_map, to_device = "auto", None
        max_mem = {i: "80GB" for i in range(torch.cuda.device_count())}

    model = model_cls.from_pretrained(
        args.path,
        trust_remote_code=True,
        device_map=device_map,
        max_memory=max_mem,
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
    )
    if to_device:
        model = model.to(to_device)
    model.eval()

    if tokenizer.pad_token_id is None:
        model.resize_token_embeddings(len(tokenizer))

    print(f"Loaded model: {model.config._name_or_path}")
    print("Model cfg override:")
    print(json.dumps(cfg, indent=2, default=lambda o: repr(o)))

    first_device = to_device if args.single_gpu else next(iter(model.hf_device_map.values()))

    # ---------- Data ----------
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(ds["text"]), add_special_tokens=False, return_tensors="pt").input_ids[0]
    if tokenizer.bos_token_id is not None:
        enc = torch.cat([torch.tensor([tokenizer.bos_token_id]), enc])

    max_len = cfg["max_len"]
    total_tokens = enc.size(0)
    nsamples = total_tokens // max_len
    if nsamples < 1:
        raise ValueError("Not enough tokens for one block of length {}".format(max_len))

    ids = enc[: nsamples * max_len].view(nsamples, max_len)

    # ---------- Batch‑size autotune ----------
    if args.batch_size == "auto":
        print("Auto‑tuning batch_size…")
        best_bs, bs = 1, 1
        while bs <= 64:
            if try_batch(model, ids, bs, first_device, pad_id):
                best_bs = bs
                bs *= 2
            else:
                bs //= 2
                break
        batch_size = best_bs
    else:
        batch_size = int(args.batch_size)
    print(f"Fixed‑block eval: blocks={nsamples}, block_size={max_len}, batch_size={batch_size}")

    # ---------- Perplexity ----------
    nll, tok_cnt = 0.0, 0
    for i in range(0, nsamples, batch_size):
        j = min(i + batch_size, nsamples)
        inp = ids[i:j].to(first_device)
        tgt = inp.clone(); tgt[:, :-1] = -100  # mask first token
        attn_mask: Optional[torch.Tensor] = None
        if is_llama4:
            attn_mask = torch.ones_like(inp).to(first_device)
        with torch.no_grad():
            loss = model(inp, attention_mask=attn_mask, labels=tgt, use_cache=False).loss.item()
        tokens = (j - i) * (max_len - 1)
        nll += loss * tokens
        tok_cnt += tokens

    ppl = math.exp(nll / tok_cnt)
    print(f"WikiText‑2 perplexity: {ppl:.2f}")
    elapsed = time.time() - start_time
    print(f"⏱️  Runtime: {elapsed:.2f} s ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()

