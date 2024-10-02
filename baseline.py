import os
import sys
import yaml
import json
import math
import copy

import numpy as np
import pandas as pd

import argparse
from models import HuggingfaceLLM, HuggingfacevLLM
from functools import partial
from tqdm import tqdm
from tools import make_prompt, make_rag_prompt

sys.path.append("clone/NeMo-Skills")

PROMPT_YAML_PATH = "clone/NeMo-Skills/nemo_skills/inference/prompt/"

DATA_ROOT_PATH = "datasets"
SAVE_ROOT_PATH = "results"
CORPUS_ROOT_PATH = "datasets/store"

from tools import EXAMPLE_TYPE


SYS = "Make sure to put the answer (and only answer) inside \\boxed{}. If it is a multi-choice question, only put (X) into it, where X is the option."
SELECT = "Select the best reasoning type for the question: {question}. You should return the reasoning type in the format of [Deductive], [Inductive], [Analogical], [Abductive] or [None]. If you think the question is not well-formed, you can return [None]. For example, if you think the question is a deductive reasoning question, you should return [Deductive]."



def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='logicqa')
    parser.add_argument('--model', type=str, default='mistral_chat')
    parser.add_argument('--mode', type=str, default="zeroshot", choices=["zeroshot","fewshot","select", "moe","icl", "rag", "def", "single"]+EXAMPLE_TYPE)
    parser.add_argument('--repeat_num', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--prompt_type', type=str, default='base')
    parser.add_argument('--problem', type=str, default='logic', choices=['logic', 'math'])
    parser.add_argument('--examples_type', type=str, default='deductive', choices=EXAMPLE_TYPE)
    parser.add_argument('--load_prompt_from', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--num_few_shots', type=int, default=3)
    parser.add_argument('--max_num', type=int, default=10)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--reset', action='store_true', default=False)
    parser.add_argument("--rag", action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    if args.problem == "logic":
        assert args.benchmark in ["logicqa", "bbh"], f"Wrong problem type for {args.benchmark}"
    elif args.problem == "math":
        assert args.benchmark in ["gsm8k", "math"], f"Wrong problem type for {args.benchmark}"
    return args

def load_file(benchmark, version="test"):
    file_path = f"{DATA_ROOT_PATH}/{benchmark}.{version}.jsonl"
    data = [json.loads(x) for x in open(file_path)]
    return data


def reload_result(save_file, reset=True):
    if not reset and os.path.exists(save_file):
        results = [x for x in open(save_file)]
        print(f"Load {len(results)} results from {save_file}")
    else:
        results = []
    return results


if __name__ == '__main__':
    args = get_options()

    if not os.path.exists(SAVE_ROOT_PATH):
        os.makedirs(SAVE_ROOT_PATH)

    data = load_file(args.benchmark)
    if args.max_num > 0:
        print(f"Keep {args.max_num} examples from {args.benchmark}...")
        data = data[:args.max_num]

    if args.mode == "zeroshot":
        args.examples_type = 'empty'
    elif args.mode == "fewshot":
        args.examples_type = 'text'
    elif args.mode == "moe":
        args.examples_type = 'all'
    elif args.mode == "def":
        args.examples_type = 'definition'
    elif args.mode == "select":
        args.examples_type = 'select'
    elif args.mode == "icl":
        args.examples_type = 'load'
    elif args.mode in EXAMPLE_TYPE:
        args.examples_type = args.mode
    print(f"[INFO] Mode: {args.mode} with {args.repeat_num} trials, prompt_type: {args.examples_type}")
    
    if args.rag:
        assert args.examples_type in EXAMPLE_TYPE + ['all'], f"RAG only supports example_type in {EXAMPLE_TYPE}"
        data = make_rag_prompt(data, args.benchmark, args.problem, args.examples_type, prompt_type=args.prompt_type)
        save_file = f'{SAVE_ROOT_PATH}/{args.benchmark}.{args.mode}.rag.{args.repeat_num}.jsonl'
    else:
        data = make_prompt(data, args.problem, args.mode, args.examples_type, prompt_type=args.prompt_type)
        save_file = f'{SAVE_ROOT_PATH}/{args.benchmark}.{args.mode}.{args.repeat_num}.jsonl'


    chunk_size = int(args.batch_size / args.repeat_num)
    results = reload_result(save_file, reset=args.reset)
    start_idx = len(results)
    print(f'[INFO] Start from idx={start_idx} ...')

    if start_idx == len(data):
        print(f'[INFO] All done!')
        sys.exit(0)

    # llm = HuggingfaceLLM(model_name=args.model)
    llm = HuggingfacevLLM(model_name=args.model)
    if args.mode != 'select':
        system_prompt = SYS
    else:
        system_prompt = ''
    generator = partial(llm.inference, batch_size=args.batch_size, stop="\n\n\n\n\n\n", temperature=args.temperature, n=args.repeat_num, sys=system_prompt, verbose=args.verbose)
    
    with open(save_file, 'w') as fout:
        for d in results:
            fout.write(d)
        fout.flush()

        for i in tqdm(range(start_idx, len(data), chunk_size), total=math.ceil((len(data)-start_idx) / chunk_size)):
            batch_data = data[i: i + chunk_size]
            batch_prompts = [x['prompt'] for x in batch_data]
            batch_results = generator(batch_prompts)
            for d, r in zip(batch_data, batch_results):
                d['response'] = r
                fout.write(json.dumps(d) + '\n')
            fout.flush()
    print(f"Save to {save_file}")


    
    


    

