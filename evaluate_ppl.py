import argparse
import torch
from tqdm import tqdm
import random
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM

def eval_ppl(model, testenc, seq_len, bs=1, device=None):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seq_len
    nlls = []
    for i in tqdm(range(0,nsamples,bs),desc="Evaluating PPL"):
        j = min(i+bs, nsamples)
        inputs = testenc[:,(i * seq_len):(j * seq_len)]
        inputs = inputs.reshape(j-i, seq_len)

        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        neg_log_likelihood = loss.float() * seq_len * (j-i)


        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_len))
    torch.cuda.empty_cache()

    return ppl.item()

def get_loaders(tokenizer, seed, seq_len=2048, nsamples=128):
    
    train_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    trainenc = tokenizer(" ".join(train_data['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(test_data['text']), return_tensors='pt')
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    return trainloader, testenc





def main(args):
    # 1) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        args.path,
        use_fast=False,
        add_bos_token=False,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        device_map="auto",
        trust_remote_code=True
    )

    # 2) Prepare test data
    _, testenc = get_loaders(tokenizer, seed=42, seq_len=args.seqlen)

    # 3) Evaluate perplexity
    ppl = eval_ppl(
        model=model,
        testenc=testenc,
        seq_len=args.seqlen,
        bs=1,
        device=args.device
    )

    # 4) Report
    print(f"Perplexity = {ppl:.2f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--path", type=str, required=True, help='model checkpoint location')
    args = parser.parse_args()
    main(args)
    