import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer,AutoModelForCausalLM

tasks = ['sst2', 'qqp', 'mnli', 'qnli', 'mnli-mm', 'rte']

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}


def main(args):
    with open(args.data_file) as f:
        dataset = json.load(f)
    model, tokenizer = load_model_tokenizer(args)
    eval(model, tokenizer, dataset, args)

def load_model_tokenizer(args):

    tokenizer = AutoTokenizer.from_pretrained(args.path,use_fast=False,add_bos_token=False,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.path,device_map="auto",trust_remote_code=True)

    return model, tokenizer
def format_example(task_name, question, origin=False, k=5):
    prompt = ""
    for i in range(k):
        if task_name in ["mnli","mnli-mm"]:
            prompt += gen_prompt(task_name, question[i], origin=origin)
            if question[i]['label'] == 0:
                prompt += " {}\n\n".format("A. yes")
            elif question[i]['label'] == 1:
                prompt += " {}\n\n".format("B. maybe")
            else:
                prompt += " {}\n\n".format("C. no")
        elif task_name in  ['qnli','rte','qqp']:
            prompt += gen_prompt(task_name, question[i], origin=origin)
            
            if question[i]['label'] == 0:
                prompt += " {}\n\n".format("A. yes")
            else:
                prompt += " {}\n\n".format("B. no")
        elif task_name =='sst2':
            prompt += gen_prompt(task_name, question[i], origin=origin)
            
            if question[i]['label'] == 0:
                prompt += " {}\n\n".format("A. positive")
            else:
                prompt += " {}\n\n".format("B. negative")
        else:
            raise ValueError("Unsupported task:", task_name)
    return prompt

def gen_prompt(task_name, question, origin=False):
    if task_name == "mnli":
        prompt = "Please identify whether the premise entails the hypothesis. The answer should be exactly 'A. yes', 'B. maybe' or 'C. no'\n"
        if origin and 'original_premise' in question.keys():
            prompt += "Premise: " + question['original_premise']
        else:
            prompt += "Premise: " + question['premise']
        prompt += "\nHypothesis: " + question['hypothesis']
        prompt += "\nAnswer: "
    elif task_name == "mnli-mm":
        prompt = "Please identify whether the premise entails the hypothesis. The answer should be exactly 'A. yes', 'B. maybe' or 'C. no'\n"
        prompt += "Premise: " + question['premise']
        if origin and 'original_hypothesis' in question.keys():
            prompt += "\nHypothesis: " + question['original_hypothesis']
        else:
            prompt += "\nHypothesis: " + question['hypothesis']
        prompt += "\nAnswer: "
    elif task_name == 'qnli':
        prompt = "Please identify whether the sentence answers the question. The answer should be exactly 'A. yes' or 'B. no'\n"
        if origin and 'original_question' in question.keys():
            prompt += "Question: " + question['original_question']
        else:
            prompt += "Question: " + question['question']
        prompt += "\nSentence: " + question['sentence']
        prompt += "\nAnswer: "
    elif task_name == 'rte':
        prompt = "Please identify whether the sentence1 entails the sentence2. The answer should be exactly 'A. yes' or 'B. no'\n"
        if origin and 'original_sentence1' in question.keys():
            prompt += "Sentence 1: " + question['original_sentence1']
        else:
            prompt += "Sentence 1: " + question['sentence1']
        prompt += "\nSentence 2: " + question['sentence2']
        prompt += "\nAnswer: "
    elif task_name == 'qqp':
        prompt = "Please identify whether Question 1 has the same meaning as Question 2. The answer should be exactly 'A. yes' or 'B. no'\n"
        # prompt = "Please identify whether the question1 entails the question2. The answer should be exactly 'A. yes' or 'B. no'\n\n"
        if origin and 'original_question1' in question.keys():
            prompt += "Question 1: " + question['original_question1']
        else:
            prompt += "Question 1: " + question['question1']
        prompt += "\nQuestion 2: " + question['question2']
        prompt += "\nAnswer: "
    elif task_name =='sst2':
        prompt = "For each snippet of text, label the sentiment of the text as positive or negative. The answer should be exactly 'A. positive' or 'B. negative'\n"
        # prompt = "Please identify whether the sentence is positive or negative. The answer should be exactly 'A. positive' or 'B. negative'\n\n"
        if origin and 'original_sentence' in question.keys():
            prompt += "Sentence: " + question['original_sentence']
        else:
            prompt += "Sentence: " + question['sentence']
        prompt += "\nAnswer: "
    else:
        raise ValueError("Unsupported task:", task_name)
    
    return prompt


def eval(model, tokenizer, dataset, args):
    cors = []
    for task_name in tasks:
        task_cors = []
        test = dataset[task_name]
        for i in range(args.ntrain, len(test)):
            prompt_end = gen_prompt(task_name, test[i], origin=args.test_origin)
            example = format_example(task_name, test, origin=args.test_origin, k=args.ntrain)
            prompt = example + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
   
            label = test[i]["label"]
            if task_name in ["mnli", "mnli-mm"]:
                logits = model(input_ids=input_ids).logits[:,-1].flatten()
                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(
                            [
                                logits[tokenizer("A").input_ids[-1]],
                                logits[tokenizer("B").input_ids[-1]],
                                logits[tokenizer("C").input_ids[-1]],
                            ]
                        ).float(),
                        dim=0,
                    )
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )
                pred = np.argmax(probs)
            else:
                logits = model(input_ids=input_ids).logits[:,-1].flatten()
                task_mappings = {
                    'qqp': {0: 1, 1: 0},
                    'sst2': {0: 1, 1: 0},
                    'qnli': {0:0, 1: 1},
                    'rte': {0:1, 1: 0}
                    }
                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(
                            [
                                logits[tokenizer("A").input_ids[-1]],
                                logits[tokenizer("B").input_ids[-1]]
                            ]
                        ).float(),
                        dim=0,
                    )
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )
                task_map = task_mappings[task_name]
                pred = task_map[np.argmax(probs)]

            cor = pred == label
            task_cors.append(cor)
            cors.append(cor)
        task_acc = np.mean(task_cors)
        print("Accuracy {:.4f} - Task {}".format(task_acc, task_name))
    
    acc = np.mean(cors)
    print("Average accuracy {:.4f}".format(acc))

def eval_generate(model, tokenizer, dataset, args):
    cors = []
    for task_name in tasks:
        task_cors = []
        test = dataset[task_name]
        for i in range(len(test)):
            prompt = gen_prompt(task_name, test[i], origin=args.test_origin)
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

            label = test[i]["label"]
            if task_name in ["mnli", "mnli-mm"]:
                logits = model(input_ids=input_ids).logits[:,-1].flatten()
                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(
                            [
                                logits[tokenizer("A").input_ids[-1]],
                                logits[tokenizer("B").input_ids[-1]],
                                logits[tokenizer("C").input_ids[-1]],
                            ]
                        ).float(),
                        dim=0,
                    )
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )
                pred = np.argmax(probs)
            else:
                logits = model(input_ids=input_ids).logits[:,-1].flatten()
                task_mappings = {
                    'qqp': {0: 1, 1: 0},
                    'sst2': {0: 1, 1: 0},
                    'qnli': {0:0, 1: 1},
                    'rte': {0:1, 1: 0}
                    }
                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(
                            [
                                logits[tokenizer("A").input_ids[-1]],
                                logits[tokenizer("B").input_ids[-1]]
                            ]
                        ).float(),
                        dim=0,
                    )
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )
                task_map = task_mappings[task_name]
                pred = task_map[np.argmax(probs)]
                cor = pred == label
                task_cors.append(cor)
                cors.append(cor)
        task_acc = np.mean(task_cors)
        print("Accuracy {:.4f} - Task {}".format(task_acc, task_name))
    
    acc = np.mean(cors)
    print("Average accuracy {:.4f}".format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5, help='number of shots')
    parser.add_argument("--path", type=str, required=True, help='model checkpoint location')
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--data_file",type=str, default='data/adv_glue/dev_ann.json', help='Input data JSON file.')
    parser.add_argument("--test_origin", action='store_true', help='Whether to test on original GLUE data.')
    args = parser.parse_args()
    main(args)