from transformers import AutoTokenizer,AutoModelForCausalLM

import argparse

from calflops import calculate_flops

def main(args):
    # load your model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.path,use_fast=False,add_bos_token=False,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.path,device_map="auto",trust_remote_code=True)

    batch_size = 1
    max_seq_length = 128
    flops, macs, params = calculate_flops(model=model,
                                        input_shape=(batch_size, max_seq_length),
                                        transformer_tokenizer=tokenizer,
                                        output_precision=2)
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help='model checkpoint location')
    args = parser.parse_args()
    main(args)