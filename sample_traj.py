import os
import sys
import yaml
import json

import numpy as np
import pandas as pd

import argparse
from models import HuggingfaceLLM, HuggingfacevLLM
from functools import partial
from tqdm import tqdm


DATA_ROOT_PATH = "dataset/sample_prompt"
SAVE_ROOT_PATH = "dataset/traj"

SYS = "Make sure to put the answer (and only answer) inside \\boxed{}. If it is a multi-choice question, only put (X) into it, where X is the option."


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default='logicqa.deductive.jsonl')
    parser.add_argument('--model', type=str, default='mistral_chat')
    parser.add_argument('--repeat_num', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--max_num', type=int, default=10)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    return args


def reload_result(save_file, reset=True):
    if not reset and os.path.exists(save_file):
        results = [x for x in open(save_file)]
        print(f"Load {len(results)} results from {save_file}")
    else:
        results = []
    return results

# python inference.py -i temp.jsonl 

if __name__ == '__main__':
    args = get_options()

    if not os.path.exists(SAVE_ROOT_PATH):
        os.makedirs(SAVE_ROOT_PATH)

    input_file = f'{DATA_ROOT_PATH}/{args.i}'
    data = [json.loads(x) for x in open(input_file)]
    if args.max_num > 0:
        data = data[:args.max_num]

    chunk_size = int(args.batch_size / args.repeat_num)
    save_file = f'{SAVE_ROOT_PATH}/{args.i}'
    results = reload_result(save_file, reset=args.reset)
    start_idx = len(results)
    print(f'[INFO] Start from idx={start_idx} ...')

    llm = HuggingfacevLLM(model_name=args.model)
    generator = partial(llm.inference, batch_size=args.batch_size, stop="\n\n\n\n\n\n", temperature=1.0, n=args.repeat_num, sys=SYS, verbose=args.verbose)
    
    with open(save_file, 'w') as fout:
        for d in results:
            fout.write(d)
        fout.flush()

        for i in tqdm(range(start_idx, len(data), chunk_size), total=(len(data)-start_idx) // chunk_size):
            batch_data = data[i: i + chunk_size]
            batch_prompts = [x['prompt'] for x in batch_data]
            batch_results = generator(batch_prompts)
            for d, r in zip(batch_data, batch_results):
                d['response'] = r
                fout.write(json.dumps(d) + '\n')
            fout.flush()


    
    


    

