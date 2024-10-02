import os
import sys
import yaml
import json
import re

import numpy as np
import pandas as pd

import argparse

sys.path.append('clone/NeMo-Skills')

from nemo_skills.inference.prompt.utils import prompt_types, Prompt, PromptConfig, FewShotExamples
from dataclasses import asdict, dataclass, field

from nemo_skills.code_execution.math_grader import math_equal
from tools import last_boxed_only_string, remove_boxed, most_frequent, get_box_option

from models import HuggingfaceLLM, HuggingfacevLLM
from functools import partial

from tools import DEFINITIONS, EXAMPLE_TYPE

PROMPT_YAML_PATH = "clone/NeMo-Skills/nemo_skills/inference/prompt/"
DATA_ROOT_PATH = "datasets/"
PROMPT_ROOT_PATH = "datasets/sample_prompt"
TRAJ_ROOT_PATH = "datasets/traj"


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mistral_chat')
    parser.add_argument('--mode', type=str, default='make_prompt', choices=['make_prompt', 'filter_traj'])
    parser.add_argument('--benchmark', type=str, default='logicqa')
    parser.add_argument('--prompt_type', type=str, default='base')
    parser.add_argument('--problem', type=str, default='logic')
    parser.add_argument('--examples_type', type=str, default='deductive')
    parser.add_argument('--num_few_shots', type=int, default=3)
    parser.add_argument('--max_num', type=int, default=-1)
    args = parser.parse_args()
    print(args)
    return args


def load_file(benchmark, version='test'):
    file_path = f"{DATA_ROOT_PATH}/{benchmark}.{version}.jsonl"
    data = [json.loads(x) for x in open(file_path)]
    return data


def do_prompt(args):
    if not os.path.exists(PROMPT_ROOT_PATH):
        os.makedirs(PROMPT_ROOT_PATH)

    data = load_file(args.benchmark, version='train')

    file = f"{PROMPT_YAML_PATH}/{args.prompt_type}.yaml"
    prompt_config_file = yaml.safe_load(open(file))
    few_show_config = prompt_config_file['few_shot_examples']


    if args.examples_type != 'all':
        examples_type_list = [args.examples_type]
    else:
        examples_type_list = EXAMPLE_TYPE


    for examples_type in examples_type_list:
        prompt_config_file['few_shot_examples'] = FewShotExamples(
                                                    examples_type=args.problem + '_' + examples_type, 
                                                    num_few_shots=args.num_few_shots, 
                                                    **few_show_config)
        prompt_config = PromptConfig(**prompt_config_file)


        save_file = f"{PROMPT_ROOT_PATH}/{args.benchmark}.{examples_type}.jsonl"
        with open(save_file, 'w') as fout:
            for d in data:
                d['prompt_type'] = prompt_config.few_shot_examples.examples_type
                if 'retrieval' not in d:
                    d['retrieval'] = ''
                if d['prompt_type'] == 'math_analogical':
                    p = Prompt(prompt_config, d, example_dicts=None, context_template="Use analogical reasoning based on a similar question and its solution:\n{retrieval}\n\n")
                else:
                    p = Prompt(prompt_config, d, example_dicts=None)
                d['prompt'] = str(p).replace('Use analogical reasoning based on a similar question and its solution:\n\n','')
                fout.write(json.dumps(d) + '\n')
        print(f"Save {len(data)} prompts to {save_file}")






def do_traj(args):
    for t in EXAMPLE_TYPE:
        file_path = f'{TRAJ_ROOT_PATH}/{args.benchmark}.{t}.jsonl'
        print(f'Read from {file_path}')
        df = pd.read_json(file_path, orient='records', lines=True)
        if args.max_num > 0:
            df = df.head(args.max_num)
        print(f'Read {len(df)} examples from {file_path}')

        df['response'] = df['response'].apply(lambda x:[x] if isinstance(x, str) else x)
        if args.problem == 'logic':
            df['response_ans'] = df['response'].apply(lambda x: [get_box_option(c) for c in x])   
            df['response_major_vote'] = df['response_ans'].apply(most_frequent)
            df['correct'] = (df['expected_answer'] == df['response_major_vote']).astype(int)
        elif args.problem == 'math':
            df['response_ans'] = df['response'].apply(lambda x: [remove_boxed(last_boxed_only_string(c)) for c in x])   
            df['response_major_vote'] = df['response_ans'].apply(most_frequent)
            df['correct'] = df.apply(lambda x: math_equal(x['response_major_vote'], x['expected_answer']), axis=1)
            df['correct'] = df['correct'].astype(int)

        print(f"INSTANCE LEVEL | Accuracy (after majority vote) {df['correct'].mean():.4f}")

        success = []
        for idx, row in df.iterrows():
            for i, ans in enumerate(row['response_ans']):
                if ans is None:
                    correct_flag = False
                if args.problem == 'logic':
                    correct_flag = (ans == row['expected_answer'])
                else:
                    correct_flag = math_equal(ans, row['expected_answer'])
                if correct_flag:
                    ins = row[['id','question','expected_answer', 'benchmark', 'task', 'prompt']].to_dict()
                    ins['response'] = row['response'][i]
                    success.append(ins)
        
        all_ans = df['response_ans'].sum()
        non_ans = [x for x in all_ans if x]
        print(f'RESPONSE LEVEL | Correct {len(success)}, Total {len(all_ans)}, Accuracy {len(success)/len(all_ans):.4f}')
        print(f'RESPONSE LEVEL | Extract {len(non_ans)}, Extract Ratio {len(non_ans) / len(all_ans):.4f}')
        
        save_file = file_path.replace('.jsonl', '.scc.jsonl')
        print(f"Save {len(success)} to {save_file}")
        with open(save_file, 'w') as fout:
            for d in success:
                fout.write(json.dumps(d) + '\n')


def do_filter(args):
    template = """Question: {question}\nSolution: {response}\nExpected Answer: {expected_answer}\nReasoning Type: {t} is {definition}\nCheck whether the Solution belongs to the Reasoning Type. Return True if it belongs, otherwise False."""

    llm = HuggingfacevLLM(model_name=args.model)
    generator = partial(llm.inference, batch_size=1000, stop="\n\n\n\n\n\n", temperature=0, n=1, verbose=True)
    

    for t in EXAMPLE_TYPE:
        if t == 'text':
            continue
        file_path = f'{TRAJ_ROOT_PATH}/{args.benchmark}.{t}.scc.jsonl'
        print(f'Read from {file_path}')
        df = pd.read_json(file_path, orient='records', lines=True)
        if args.max_num > 0:
            df = df.head(args.max_num)
        print(f'Read {len(df)} examples from {file_path}')

        prompts = []
        for idx, row in df.iterrows():
            row['t'] = t
            row['definition'] = DEFINITIONS[t]
            prompt = template.format(**row)
            prompts.append(prompt)
        
        responses = generator(prompts)
        df['response'] = responses

        save_file = file_path.replace('.jsonl', '.filter.jsonl')
        print(f"Save {len(df)} to {save_file}")
        df.to_json(save_file, orient='records', lines=True)


# python sample_helper.py --mode make_prompt --benchmark logicqa 
# python sample_helper.py --mode make_prompt --benchmark gsm8k --problem math 
# python sample_helper.py --mode make_traj --benchmark logicqa 
# python sample_helper.py --mode make_traj --benchmark gsm8k --problem math 
# python sample_helper.py --mode filter_traj --benchmark logicqa 
# python sample_helper.py --mode filter_traj --benchmark math --problem math 

if __name__ == '__main__':
    args = get_options()

    if args.mode == 'make_prompt':
        do_prompt(args)
    elif args.mode == 'make_traj':
        do_traj(args)
    elif args.mode == 'filter_traj':
        do_filter(args)
    else:
        raise NotImplementedError
    

    

