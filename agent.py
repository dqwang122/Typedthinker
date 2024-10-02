
import os
import re
import json
import argparse

from tqdm import tqdm
from typing import Iterable, List, Dict

import numpy as np

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model


from models import MODEL_NAME_PATH, HuggingfaceLLM, HuggingfacevLLM
from tools import get_policy, make_prompt, make_rag_prompt

MODEL_NAME_OR_PATH = "mistralai/Mistral-7B-Instruct-v0.2"

from tools import DEFINITIONS, EXAMPLE_TYPE

MAX_TARGET_LENGTH = 512
DATA_VERSION = "all"

def get_last_checkpoint_step(model_dir):
    files = os.listdir(model_dir)
    files = [x for x in files if x.startswith("checkpoint-")]
    files = [x.split("-")[-1] for x in files]
    steps = [int(re.sub(r"[^0-9]", "", x)) for x in files]
    steps = sorted(steps)
    return steps[-1]

class ReasoningAgent():
    def __init__(self, model_name: str = "mistral_chat", padding_side: str = "left"):
        super().__init__()

        self.backend = None
        self.tokenizer = None

        self.model_name = model_name
        self.model_name_or_path = MODEL_NAME_PATH.get(model_name, model_name)
        self.padding_side = padding_side

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side=padding_side,
            use_fast=False
        )
        self.tokenizer.padding_side = padding_side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def init(self):
        self.backend = HuggingfacevLLM(model_name=self.model_name, padding_side=self.padding_side)

    def prepare_for_finetune(self, lora=False):
        # self.backend.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        # model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, torch_dtype=torch.bfloat16)
        model.config.use_cache = False
        if lora:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, peft_config)
        model.train()
        return (model, self.tokenizer)

    def inference(self, prompts: List, batch_size=1, max_tokens=1000, n=1, verbose=False, **kwargs) -> Iterable[str]:
        if self.backend is None:
            self.init()
        return self.backend.inference(prompts, batch_size=batch_size, max_tokens=max_tokens, n=n, verbose=verbose, **kwargs)

    def get_chat_format(self, m, tokenizer=None):
        x = tokenizer.apply_chat_template(m, tokenize=False)
        return x

    def get_reasoning_type(self, data: List[Dict], batch_size=1, max_tokens=1000, n=1, verbose=False, **kwargs):
        SELECT = '''Given the question below, please identify the type of reasoning required to provide a solution. You may choose from the following reasoning types: Deductive, Inductive, Analogical, Abductive, or None. None indicates that no specific reasoning type is needed for this problem. For each reasoning type selected, please assign a confidence score from 0 to 1, where 0 represents no confidence and 1 represents full confidence. Please return the reasoning types and their corresponding confidence scores in the JSON format. \n\nFor instance, if you think the question can be solved using both deductive and inductive reasoning, with a confidence of 0.5 for deductive reasoning and 0.3 for inductive reasoning, you should return:\n[{{"ReasoningType": "Deductive", "Confidence": 0.5}},{{"ReasoningType": "Inductive", "Confidence": 0.3}},{{"ReasoningType": "Analogical", "Confidence": 0}},{{"ReasoningType": "Abductive", "Confidence": 0}}, {{"ReasoningType": "None", "Confidence": 0}}]\n\nQuestion: {question}'''.strip()
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        for d in data:
            prompt = SELECT.format(question=d['question'])
            chat_format = self.get_chat_format([{"role": "user", "content": prompt}], tokenizer=tokenizer)
            prompts.append(chat_format)
        
        response = self.inference(prompts, batch_size=batch_size, max_tokens=max_tokens, n=1, verbose=verbose, chat_format=False, **kwargs)
        policys = [get_policy(r) for r in response]
        return policys

    def get_finetuned_reasoner(self, data: List[Dict], batch_size=1, max_tokens=1000, n=1, verbose=False, **kwargs):
        TEMPLATE = """Use {reasoning} Reasoning to solve the given question. {reasoning} Reasoning is {definition}\n\nQuestion: {question}\n\nMy solution:\n"""
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        prompts = []
        for d in data:
            for t in DEFINITIONS.keys():
                prompt = TEMPLATE.format(reasoning=t.capitalize(), DEFINITIONS=DEFINITIONS[t], question=d['question'])
                chat_format = self.get_chat_format([{"role": "user", "content": prompt}], tokenizer=tokenizer)
                prompts.append(chat_format)

        response = self.inference(prompts, batch_size=batch_size, max_tokens=max_tokens, n=1, chat_format=False, verbose=verbose, **kwargs)
        return response
        
    def apply_policy(self, data: List[Dict], mode: str, policys: List[str], policy_mode='best', use_rag=False, batch_size=1, n=1, verbose=False, **kwargs):
        if self.backend is None:
            self.init()

        if 'benchmark' not in data[0]:
            benchmark = "math"
        else:
            benchmark = data[0]['benchmark']
        
        if benchmark in ["logicqa", "bbh"]:
            problem = "logic"
        else:
            problem = "math"

        policy_type_group = {}
        for idx, (d, p) in enumerate(zip(data, policys)):
            if policy_mode == 'best':
                policy_type = max(p, key=p.get)
                if policy_type not in policy_type_group:
                    policy_type_group[policy_type] = [d]
                else:
                    policy_type_group[policy_type].append(d)
            elif policy_mode == 'weighted':
                policy_type = [x for x in p.keys() if p[x] != 0]
                policy_type = ["text" if x == "none" else x for x in policy_type]
                for p in policy_type:
                    if p not in policy_type_group:
                        policy_type_group[p] = [d]
                    else:
                        policy_type_group[p].append(d)
            else:
                raise NotImplementedError

        # policy_type_group['text'] = policy_type_group.pop('none')
        new_data = []
        for p in policy_type_group:
            print(f"Type: {p}, Size: {len(policy_type_group[p])}")
            if use_rag:
                nd = make_rag_prompt(policy_type_group[p], benchmark=benchmark, problem=problem, examples_type=p)
            else:
                nd = make_prompt(policy_type_group[p], problem=problem, mode=mode, examples_type=p)
            new_data.extend(nd)

        prompts = [d['prompt'] for d in new_data]
        response = self.inference(prompts, batch_size=batch_size, n=n, verbose=verbose, **kwargs)

        for d, r in zip(new_data, response):
            d['response'] = r
        
        return new_data

    
def infer_valid_policy(args):
    step = args.step
    model_version = args.train_from
    data_version = DATA_VERSION
    model_name = f"checkpoints/{model_version}/checkpoint-{step}"
    device = "cuda:0"

    agent = ReasoningAgent(model_name)

    file = f"datasets/{data_version}.policy.valid.json"
    data = json.load(open(file))
    instances = data['instances']
    prompts = [ins['input'] for ins in instances]
    print(f"Read {len(prompts)} examples from {file}")

    response = agent.inference(prompts, batch_size=100, chat_format=False)

    save_dir = f"results/{model_version}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = f"{save_dir}/{data_version}.policy.valid.step{step}.jsonl"
    with open(save_file, 'w') as fout:
        for ins, rep in zip(instances, response):
            ins['response'] = rep
            fout.write(json.dumps(ins) + '\n')
    print(f"Save {len(prompts)} examples to {save_file}")

def infer_valid_sft(args):
    step = args.step
    model_version = args.train_from
    model_name = f"checkpoints/{model_version}/checkpoint-{step}"

    agent = ReasoningAgent(model_name)

    data_version = DATA_VERSION
    file = f"datasets/{data_version}.sft.valid.json"
    data = json.load(open(file))
    instances = data['instances']
    prompts = [ins['input'] for ins in instances]
    print(f"Read {len(prompts)} examples from {file}")

    response = agent.inference(prompts, batch_size=100, chat_format=False, verbose=True)

    save_dir = f"results/{model_version}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = f"{save_dir}/{data_version}.sft.valid.step{step}.jsonl"
    with open(save_file, 'w') as fout:
        for ins, rep in zip(instances, response):
            ins['response'] = rep
            fout.write(json.dumps(ins) + '\n')
    print(f"Save {len(prompts)} examples to {save_file}")

def infer_test_policy(args):
    step = args.step
    model_version = args.train_from
    model_name = f"checkpoints/{model_version}/checkpoint-{step}"
    device = "cuda:0"

    agent = ReasoningAgent(model_name)

    if args.benchmark == 'all':
        benchmark_list = ["logicqa", "bbh", "gsm8k", "math"]
    else:
        benchmark_list = [args.benchmark]

    for benchmark in benchmark_list:
        file = f"datasets/{benchmark}.test.jsonl"
        data = [json.loads(x) for x in open(file)]
        print(f"Read {len(data)} examples from {file}")

        response = agent.get_reasoning_type(data, batch_size=1000, verbose=True)

        save_dir = f"results/{model_version}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = f"{save_dir}/{benchmark}.test.step{step}.jsonl"
        with open(save_file, 'w') as fout:
            for ins, rep in zip(data, response):
                ins['policy'] = rep
                fout.write(json.dumps(ins) + '\n')
        print(f"Save {len(response)} examples to {save_file}")

def infer_test_sft(args):
    step = args.step
    model_version = args.train_from
    model_name = f"checkpoints/{model_version}/checkpoint-{step}"
    device = "cuda:0"

    agent = ReasoningAgent(model_name)

    if args.benchmark == 'all':
        benchmark_list = ["logicqa", "bbh", "gsm8k", "math"]
    else:
        benchmark_list = [args.benchmark]

    for benchmark in benchmark_list:
        file = f"datasets/{benchmark}.test.jsonl"
        data = [json.loads(x) for x in open(file)]
        print(f"Read {len(data)} examples from {file}")

        response = agent.get_finetuned_reasoner(data, batch_size=1000, verbose=True)
        response = [response[i: i + len(DEFINITIONS)] for i in range(0, len(response), len(DEFINITIONS))]

        save_dir = f"results/{model_version}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = f"{save_dir}/{benchmark}.test.step{step}.jsonl"
        with open(save_file, 'w') as fout:
            for ins, rep in zip(data, response):
                for t, r in zip(DEFINITIONS.keys(), rep):
                    ins['type'] = t
                    ins['response'] = r
                    fout.write(json.dumps(ins) + '\n')
        print(f"Save {len(response)} examples to {save_file}")

def apply_policy(args):
    step = args.step
    model_version = args.train_from
    load_version = args.load_from

    if model_version is not None:
        model_dir = f"checkpoints/{model_version}"
        last_step = get_last_checkpoint_step(model_dir)
        model_name = f"{model_dir}/checkpoint-{last_step}"
    else:
        model_version = args.backend
        model_name = args.backend

    agent = ReasoningAgent(model_name=model_name)

    if args.benchmark == 'all':
        benchmark_list = ["logicqa", "bbh", "gsm8k", "math"]
    else:
        benchmark_list = [args.benchmark]

    for benchmark in benchmark_list:
        file = f"results/{load_version}/{benchmark}.test.step{step}.jsonl"
        data = [json.loads(x) for x in open(file)]
        print(f"Read {len(data)} examples from {file}")

        policy = [x['policy'] for x in data]
        response = agent.apply_policy(data, policys=policy, mode=args.mode, policy_mode=args.policy_mode, use_rag=args.rag, batch_size=1000, n=args.n, temperature=args.temperature, verbose=True)
        
        if args.rag:
            save_file = file.replace('.jsonl', f'.{model_version}.{args.mode}.rag.{args.n}.jsonl')
        else:
            save_file = file.replace('.jsonl', f'.{model_version}.{args.mode}.{args.n}.jsonl')
        with open(save_file, 'w') as fout:
            for ins in response:
                fout.write(json.dumps(ins) + '\n')
        print(f"Save {len(response)} examples to {save_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, default="valid")
    parser.add_argument("--backend", type=str, default="mistral_chat")
    parser.add_argument("--train_from", type=str, default=None)
    parser.add_argument("--load_from", type=str, default="sft_all_sft_2")
    parser.add_argument("--step", type=int, default=150)
    parser.add_argument("--benchmark", type=str, default="bbh")
    parser.add_argument("--mode", type=str, default="moe")
    parser.add_argument("--policy_mode", type=str, default='best', choices=['best', 'weighted'])
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--rag", action='store_true', default=False)
    parser.add_argument("--sft", action='store_true', default=False)

    args = parser.parse_args()

    if args.func == "valid":
        if args.sft:
            infer_valid_sft(args)
        else:
            infer_valid_policy(args)
    elif args.func == "test":
        if args.sft:
            infer_test_sft(args)
        else:
            infer_test_policy(args)
    elif args.func == "apply":
        apply_policy(args)
    else:
        raise NotImplementedError


    



    