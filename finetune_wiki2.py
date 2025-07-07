#!/usr/bin/env python
"""
finetune_wiki2.py – “Full-window” (4 096 Token) Finetuning
==========================================================
* lädt einen bereits distillierten Checkpoint
* erstellt 4 096-Token-Blöcke aus WikiText-2  (Train + Val)
* finetuned drei Epochen mit bfloat16 + Gradient-Checkpointing
"""
import os, math, random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    LlamaConfig,
    LlamaForCausalLM,
    TrainingArguments,
)

# --------------------------------------------------------------------------- #
# 0) Pfad-Konstanten – nur hier anpassen                                      #
# --------------------------------------------------------------------------- #
CHECKPOINT  = "/home/geiger/gwb082/Jonathans_Thesis/compressed-models/distilled/llama2_3.5b_distill_lr1e-4_wiki2_alpha0.5_seq1024/checkpoint-10000"
OUT_DIR     = "./finetuned_wiki2_seq1024"
BLOCK_SIZE  = 4_096                 # Kontextlänge beim Eval
BATCH_SIZE  = 2                     # 4 096 Token passen meist nur als 1×
ACC_STEPS   = 8                    # → eff. Batch = 16 Blöcke

# --------------------------------------------------------------------------- #
###############################################################################
# Helper functions                                                             #
###############################################################################


def chunk_wikitext(split: str):
    """Tokenisiere WikiText-2 und falte es in feste 4 096-Token-Blöcke"""
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # a) Tokenizer (shared)
    tok = tokenizer  # <- aus dem globalen Kontext

    def tokenize(batch):
        return tok(batch["text"], add_special_tokens=False)  # nur reiner Text

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])

    # b) alle Token hintereinander hängen und in Blöcke schneiden
    def group(block):
        # flatten → chunk
        all_tokens = sum(block["input_ids"], [])
        # floor division sorgt dafür, dass nur volle Blöcke benutzt werden
        total_len = (len(all_tokens) // BLOCK_SIZE) * BLOCK_SIZE
        ids = [all_tokens[i : i + BLOCK_SIZE]
               for i in range(0, total_len, BLOCK_SIZE)]
        return {"input_ids": ids}

    return tokenized.map(
        group,
        batched=True,
        num_proc=4,
        remove_columns=tokenized.column_names,
        desc=f"Chunking {split}",
    )

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.cuda.empty_cache()

    # 3) Model initialisieren
    model = LlamaForCausalLM.from_pretrained(CHECKPOINT, torch_dtype=torch.bfloat16, device_map="auto").eval()

    # 1) Tokenizer & Modell -------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token               # <- pad = eos

    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()                    # RAM ↘

    # 2) Dataset – 4 096-Token-Chunks --------------------------------------- #
    train_ds = chunk_wikitext("train")
    val_ds   = chunk_wikitext("validation")

    # 3) Collator – klassisches causal LM                                   #
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 4) TrainingArguments --------------------------------------------------- #
    args = TrainingArguments(
        output_dir           = OUT_DIR,
        overwrite_output_dir = True,
        num_train_epochs     = 3,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = ACC_STEPS,
        learning_rate        = 5e-5,
        warmup_steps         = 200,
        logging_steps        = 50,
        save_steps           = 500,
        bf16                 = True,
        eval_strategy  = "steps",
        eval_steps           = 500,
        save_total_limit     = 3,
        fp16                 = False,
        report_to            = "none",
    )

    # 5) Trainer ------------------------------------------------------------- #
    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        tokenizer       = tokenizer,
    )

    # 6) Finetuning ---------------------------------------------------------- #
    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
