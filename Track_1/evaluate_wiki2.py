#!/usr/bin/env python3
import argparse
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader


def group_texts(examples, block_size):
    # Concatenate all texts
    concatenated = sum(examples['input_ids'], [])
    total_length = len(concatenated)
    # Drop the last chunk if it's smaller than block_size
    total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size
    result = {
        'input_ids': [concatenated[i: i + block_size] for i in range(0, total_length, block_size)]
    }
    return result


@torch.no_grad()
def evaluate(model, tokenizer, dataset, args):
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: tokenizer.pad(
            {'input_ids': batch}, return_tensors='pt'
        )
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        # labels same as input_ids
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        # loss is average over all tokens
        batch_tokens = input_ids.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate causal LM on WikiText2 perplexity")
    parser.add_argument("--path", type=str, required=True,
                        help="Model checkpoint path or HF repo ID")
    parser.add_argument("--block_size", type=int, default=512,
                        help="Sequence length (block size) for evaluation")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per evaluation step")
    parser.add_argument("--torch_dtype", type=str, default="torch.float16",
                        help="Torch dtype for model weights (torch.float16 or torch.float32)")
    args = parser.parse_args()

    # Load tokenizer and model
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
    )

    # Load WikiText2 raw dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Tokenize
    tokenized = ds.map(
        lambda examples: tokenizer(examples['text'], add_special_tokens=False),
        batched=True,
        remove_columns=['text'],
    )
    # Group into blocks
    grouped = tokenized.map(
        lambda ex: group_texts(ex, args.block_size),
        batched=True,
    )
    # Use 'input_ids' column for evaluation
    dataset = grouped['input_ids'] if isinstance(grouped, dict) else grouped

    # Evaluate
    avg_loss, ppl = evaluate(model, tokenizer, dataset, args)
    print(
        f"WikiText2 evaluation -- avg_loss: {avg_loss:.4f}, perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
