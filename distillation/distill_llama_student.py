#!/usr/bin/env python
"""
distill_llama_student.py
-----------------------
Creates a student model with roughly half the parameters of a Llama‑2‑7B (~3.5 B) and
distils knowledge from the original teacher.

Example (single GPU)
--------------------
python distill_llama_student.py \
  --teacher_path meta-llama/Llama-2-7b-hf \
  --output_dir /tmp/llama2_3b_distilled

Call‑ready for SLURM via srun / sbatch with plenty of CLI flags.
"""

import argparse
import os
import math
import torch
import time
from torch.nn import KLDivLoss
from datasets import load_dataset
import transformers, inspect
# … your other imports …

from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    EvalPrediction,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.integrations import WandbCallback
import wandb
import os


print("Transformers-Version:", transformers.__version__)
print("TrainingArguments   :", inspect.getfile(TrainingArguments))


###############################################################################
# Helper functions                                                             #
###############################################################################
def build_student(teacher_cfg: LlamaConfig, layer_ratio: float = 0.5, keep_last: int = 2,) -> LlamaForCausalLM:
    """
    Create a student model that keeps:
      - the bottom `int(num_hidden_layers * layer_ratio)` layers
      - PLUS the last `keep_last` layers of the teacher
    """
    total = teacher_cfg.num_hidden_layers
    bottom = max(1, int(total * layer_ratio))
    student_layers = bottom + keep_last
    cfg_dict = teacher_cfg.to_dict()
    cfg_dict["num_hidden_layers"] = student_layers
    return LlamaForCausalLM(LlamaConfig.from_dict(cfg_dict))


def copy_layers(student: LlamaForCausalLM, teacher: LlamaForCausalLM, keep_last: int = 2,) -> None:
    """
    Copy into the student:
      - teacher.layers[0:bottom]
      - teacher.layers[-keep_last:]
      - plus embeddings & head
    """
    with torch.no_grad():
        total = teacher.config.num_hidden_layers
        bottom = student.config.num_hidden_layers - keep_last

        # 1) Copy bottom layers
        for i in range(bottom):
            student.model.layers[i].load_state_dict(
                teacher.model.layers[i].state_dict()
            )
        # 2) Copy last `keep_last` layers
        for j in range(keep_last):
            student.model.layers[bottom + j].load_state_dict(
                teacher.model.layers[total - keep_last + j].state_dict()
            )

        # 3) Embeddings & LM head as before
        student.model.embed_tokens.load_state_dict(
            teacher.model.embed_tokens.state_dict()
        )
        student.lm_head.load_state_dict(teacher.lm_head.state_dict())



def distillation_loss(student_logits, teacher_logits, temperature: float = 2.0):
    s_logits = student_logits.float()
    t_logits = teacher_logits.float()

    student_log_probs = torch.log_softmax(s_logits / temperature, dim=-1)
    teacher_probs = torch.softmax(t_logits / temperature, dim=-1)
    return KLDivLoss(reduction="batchmean")(student_log_probs, teacher_probs) * (temperature**2)

###############################################################################
# KD‑Trainer                                                                   #
###############################################################################
import csv
import math
from transformers import TrainerCallback

class CombinedPPLCallback(TrainerCallback):
    def __init__(self, csv_filename="eval_ppl_by_epoch_{args.}.csv"):
        self.csv_filename = csv_filename
        # CSV-Header anlegen
        with open(self.csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "eval_perplexity"])

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        loss = metrics.get("eval_loss", None)
        if loss is None:
            return

        # 1) compute once
        try:
            ppl = math.exp(loss)
        except OverflowError:
            ppl = float("inf")
        metrics["perplexity"] = ppl

        # 2) log to W&B (falls Du das weiter willst)
        import wandb
        wandb.log({"eval_perplexity": ppl}, step=state.global_step)

        # 3) print in stdout
        print(f"\n>>> Eval Perplexity at step {state.global_step}, epoch {metrics.get('epoch'):.2f}: {ppl:.2f}\n")

        # 4) append to CSV
        epoch = metrics.get("epoch", None)
        with open(self.csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, state.global_step, ppl])

                
class DistilTrainer(Trainer):
    """Hugging Face Trainer that mixes LM‑loss with KD‑loss."""

    def __init__(
        self,
        teacher: LlamaForCausalLM,
        temperature: float = 2.0,
        alpha_distill: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.temperature = temperature
        self.alpha = alpha_distill

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forward des Student-Modells
        outputs = model(**inputs)
        lm_loss = outputs.loss

        # Im Training zusätzlich KD-Loss, in der Eval-Phase nur LM-Loss zurückgeben
        if model.training:
            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs)
            kd_loss = distillation_loss(outputs.logits, teacher_outputs.logits, self.temperature)
            loss = self.alpha * kd_loss + (1 - self.alpha) * lm_loss
        else:
            loss = lm_loss

        return (loss, outputs) if return_outputs else loss

###############################################################################
# CLI                                                                          #
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--teacher_path", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_name", default="wikipedia")
    p.add_argument("--dataset_config", default="20220301.en")
    p.add_argument("--layer_ratio", type=float, default=0.5)
    p.add_argument("--local_rank", type=int, default=-1,
                   help="automatisch von torchrun gesetzt")
    p.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps for learning rate scheduler")
    p.add_argument("--max_seq_length", type=int, default=2048, help="Truncate examples to this length")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--wiki_pct", type=float, default=2.0, help="Prozentsatz von Wikipedia für Training (z.B. 2.0 für '[:2%]')")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha_distill", type=float, default=0.5)

    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)

    return p.parse_args()

###############################################################################
# Main                                                                         #
###############################################################################

def main():
    args = parse_args()
    start_time = time.time()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print("=== Hyperparameter ===")
    print(f"  Learning rate:                  {args.learning_rate}")
    print(f"  Warmup steps:                   {args.warmup_steps}")
    print(f"  Gradient accumulation steps:    {args.gradient_accumulation_steps}")
    print(f"  Per-device batch size:          {args.per_device_train_batch_size}")
    print(f"  Num epochs:                     {args.num_train_epochs}")
    print(f"  Temperature (T):                {args.temperature}")
    print(f"  Distillation α:                 {args.alpha_distill}")
    print(f"  Layer ratio:                    {args.layer_ratio}")
    print(f"  Max sequence length:            {args.max_seq_length}")
    print(f"  % of Wikipedia for Training:    {args.wiki_pct}%")
    print(f"  GPUs (torch.cuda.device_count): {num_gpus}")
    print("======================\n")
    os.makedirs(args.output_dir, exist_ok=True)

    # 0) WandB – optional
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # von torchrun gesetzt

    if local_rank != 0:
        # Nebenprozesse: W&B komplett ausschalten
        os.environ["WANDB_MODE"] = "disabled"

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or "llama2-7b_to_student",
        config=vars(args),
    )

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # device_map = "auto" if torch.cuda.is_available() else None

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    # 1) Teacher inF16 auf einer GPU
    teacher = LlamaForCausalLM.from_pretrained(
        args.teacher_path,
        torch_dtype=torch.bfloat16,
#        attn_implementation="flash_attention_2",
        device_map={"": device},   # alle Teacher-Gewichte auf cuda:local_rank
    ).eval()

    # 2) Student halb so viele Layer auf CPU aufbauen
    student = build_student(teacher.config, args.layer_ratio)
    #student.config.attn_implementation = "flash_attention_2"
    copy_layers(student, teacher)  # kopiert bereits float32-Gewichte
    # student.gradient_checkpointing_enable() # <-- spart Aktivierungs-RAM
    student = student.to(device, torch.bfloat16)  # auf GPU verschieben

    # --------------------------------------------------------------------- #
    # 2)  Tokenizer & Dataset                                               #
    # --------------------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=f"train[:{int(args.wiki_pct)}%]",
        trust_remote_code=True
    ).shuffle(seed=42)

    train_ds, valid_ds = raw.train_test_split(test_size=0.01, seed=42).values()

    text_field = next(col for col in ("text", "article", "content") if col in raw.column_names)

    def tokenize(batch):
        return tokenizer(batch[text_field],
                         truncation=True,
                         max_length=args.max_seq_length)

    # num_proc moderat, sonst CPU-Overload
    num_proc = min(8, os.cpu_count())
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=raw.column_names,
                             num_proc=num_proc, desc="Tokenising train")
    valid_tok = valid_ds.map(tokenize, batched=True, remove_columns=raw.column_names,
                             num_proc=max(1, num_proc // 2), desc="Tokenising valid")

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # --------------------------------------------------------------------- #
    # 3)  TrainingArguments                                                 #
    # --------------------------------------------------------------------- #
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        bf16=True,  # torch_dtype=auto
        bf16_full_eval=True,  # eval in bf16,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        eval_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb",
    )

    csv_name = f"distill_results/eval_ppl_lr{args.learning_rate}_alpha{args.alpha_distill}_wiki{int(args.wiki_pct)}.csv"

    # --------------------------------------------------------------------- #
    # 4)  Trainer                                                           #
    # --------------------------------------------------------------------- #
    trainer = DistilTrainer(
        teacher=teacher,
        temperature=args.temperature,
        alpha_distill=args.alpha_distill,
        model=student,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=None,
        callbacks=[CombinedPPLCallback(csv_filename=csv_name)],
    )

    trainer.train()

    # --------------------------------------------------------------------- #
    # 5)  Save                                                              #
    # --------------------------------------------------------------------- #
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()

    # Am Ende: Laufzeit ausgeben
    elapsed = time.time() - start_time
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n=== Distillation completed in {int(hrs)}h {int(mins)}m {secs:.1f}s ===")



if __name__ == "__main__":
    main()






