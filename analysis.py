import os
import sys
import yaml
import json
import re

import numpy as np
import pandas as pd

import argparse

sys.path.append("clone/NeMo-Skills")


from nemo_skills.code_execution.math_grader import math_equal
from tools import get_policy, last_boxed_only_string, remove_boxed, most_frequent, get_box_option, get_plain_option, get_mixed_option, calculate_correlation

PROMPT_YAML_PATH = "clone/NeMo-Skills/nemo_skills/inference/prompt/"
DATA_ROOT_PATH = "datasets"
RESULT_ROOT_PATH = "results"

EXAMPLE_TYPE = ['deductive', 'inductive', 'analogical', 'abductive', 'none']


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='do_accu')
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--mode', type=str, default="zeroshot")
    parser.add_argument('--version', type=str, default="0429")
    parser.add_argument('--repeat_num', type=int, default=1)
    parser.add_argument('--benchmark', type=str, default='logicqa')
    parser.add_argument('--prompt_type', type=str, default='base')
    parser.add_argument('--problem', type=str, default='logic')
    parser.add_argument('--examples_type', type=str, default='deductive')
    parser.add_argument('--num_few_shots', type=int, default=3)
    parser.add_argument('--max_num', type=int, default=-1)
    args = parser.parse_args()
    print(args)
    return args


def load_file(benchmark, version, split='test'):
    file_path = f"{DATA_ROOT_PATH}/{version}/{benchmark}.{split}.jsonl"
    data = [json.loads(x) for x in open(file_path)]
    return data


def agg_moe(subdf):
    all_response = subdf['response'].values.tolist()
    res = subdf.iloc[0]
    res['response'] = all_response
    res.pop('id')
    # return pd.Series({'response': all_response})
    return res

def do_accu(args):
    if args.input_file is not None:
        file_path = args.input_file
    else:
        file_path = f'{RESULT_ROOT_PATH}/{args.version}/{args.benchmark}.{args.mode}.{args.repeat_num}.jsonl'
    print(f'Read from {file_path}')
    df = pd.read_json(file_path, orient='records', lines=True)
    if args.max_num > 0:
        df = df.head(args.max_num)
    print(f'Read {len(df)} examples from {file_path}')

    if args.mode == 'moe' or args.mode == 'def':
        df = df.groupby('question').apply(func=lambda subdf: agg_moe(subdf))
    else:
        df['response'] = df['response'].apply(lambda x: [x] if isinstance(x, str) else x)

    if args.problem == 'logic':
        assert args.benchmark in ["logicqa", "bbh"], f"Wrong problem type for {args.benchmark}"
        df['response_ans'] = df['response'].apply(lambda x: [get_mixed_option(c) for c in x]) 
        df['response_major_vote'] = df['response_ans'].apply(most_frequent)
        df['correct'] = (df['expected_answer'] == df['response_major_vote']).astype(int)
    elif args.problem == 'math':
        assert args.benchmark in ["gsm8k", "math"], f"Wrong problem type for {args.benchmark}"
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
                # ins = row[['id','question','expected_answer', 'benchmark', 'task', 'prompt']].to_dict()
                ins = row.to_dict()
                ins['response'] = row['response'][i]
                success.append(ins)
    
    all_ans = df['response_ans'].sum()
    non_ans = [x for x in all_ans if x]
    print(f'RESPONSE LEVEL | Correct {len(success)}, Total {len(all_ans)}, Accuracy {len(success)/len(all_ans):.4f}')
    print(f'RESPONSE LEVEL | Extract {len(non_ans)}, Extract Ratio {len(non_ans) / len(all_ans):.4f}')

    if args.input_file is not None:
        save_file = args.input_file.replace(".jsonl", ".csv")
    else:
        save_file = f'{RESULT_ROOT_PATH}/{args.version}/{args.benchmark}.{args.mode}.{args.repeat_num}.csv'
    print(f"Save summary to {save_file}")
    df.to_csv(save_file, index=False)


def check_policy(args):
    file_path = f'{args.input_file}'
    data = pd.read_json(file_path, orient='records', lines=True)
    print(f'Read {len(data)} examples from {file_path}')

    data['output_flat'] = data['output'].apply(get_policy)
    data['output_best'] = data['output_flat'].apply(lambda x: max(x, key=x.get))
    data['response_flat'] = data['response'].apply(get_policy)
    data['response_best'] = data['response_flat'].apply(lambda x: max(x, key=x.get))

    all_match = 0
    all_total = 0
    for key in EXAMPLE_TYPE:
        print(key)
        pred = data['response_flat'].apply(lambda x:x.get(key, -1)).values
        refer = data['output_flat'].apply(lambda x:x.get(key, -1)).values
        non_empty = tuple([pred!=-1])
        results = calculate_correlation(pred[non_empty], refer[non_empty])
        for n,v in results.items():
            if isinstance(v, tuple):
                print(f"{n}\t{v[0]}")
            else:
                print(f"{n}\t{v:0.4f}")
        
        pred = (data['response_best'].values[non_empty] == key)
        refer = (data['output_best'].values[non_empty] == key)
        match = ((pred == refer) & (refer)).sum()
        pred_sum = sum(pred)
        total = sum(refer)
        all_match += match
        all_total += total
        print(f"Match {match}, Pred {pred_sum}, Refer {total}, Accu {match / total:0.4f}")
        print()
    print(f"Match {all_match}, Refer {all_total}, Accu {all_match / all_total:0.4f}")

    save_file = file_path.replace('.jsonl', '.csv')
    data.to_csv(save_file, index=False)
    print(f"Save to {save_file}")

def check_sft(args):
    file_path = f'{args.input_file}'
    data = pd.read_json(file_path, orient='records', lines=True)
    print(f'Read {len(data)} examples from {file_path}')

    data.drop_duplicates(subset=['input'], inplace=True)
    print(f"Drop duplicates, remain {len(data)} examples")

    def process(row):
        if row['benchmark'] in ['logicqa', 'bbh']:
            row['output_ans'] = get_mixed_option(row['output'])
            row['response_ans'] = get_mixed_option(row['response'])
            row['correct'] = (row['output_ans'] == row['response_ans'])
        else:
            row['output_ans'] = remove_boxed(last_boxed_only_string(row['output']))
            row['response_ans'] = remove_boxed(last_boxed_only_string(row['response']))
            row['correct'] = math_equal(row['output_ans'], row['response_ans'])
        return row  

    data = data.apply(process, axis=1)
    data['reasoning'] = data['input'].apply(lambda x: re.findall(r"Use ([A-Za-z]+) Reasoning", x)[0])
    data['correct'] = data['correct'].astype(int)

    for name, group in data.groupby('benchmark'):
        print(f"{name}, {group['correct'].mean():0.4f}")
        for key in ['Deductive', 'Inductive', 'Analogical', 'Abductive']:
            total = group[group['reasoning']==key].shape[0]
            accu = group[group['reasoning']==key]['correct'].mean()
            print(f'\t{key} | Size Accu {accu:0.4f} | {total}')

    save_file = file_path.replace('.jsonl', '.csv')
    data.to_csv(save_file, index=False)
    print(f"Save to {save_file}")

def do_weighted(args):
    file_path = f'{args.input_file}'
    df = pd.read_csv(file_path)
    print(f'Read {len(df)} examples from {file_path}')

    if args.problem == 'logic':
        df['response_weighted_vote'] = df.apply(lambda x:most_frequent(eval(x['response_ans']), weight=[w for w in eval(x['policy']).values() if w!=0]), axis=1)
        df['weighted_correct'] = (df['expected_answer'] == df['response_weighted_vote']).astype(int)
    elif args.problem == 'math':
        df['response_weighted_vote'] = df.apply(lambda x:most_frequent(eval(x['response_ans']), weight=[w for w in eval(x['policy']).values() if w!=0]), axis=1)
        df['weighted_correct'] = df.apply(lambda x: math_equal(x['response_weighted_vote'], x['expected_answer']), axis=1)
        df['weighted_correct'] = df['weighted_correct'].astype(int)

    print(f"INSTANCE LEVEL | Accuracy (after weighted vote) {df['weighted_correct'].mean():.4f}")

    save_file = args.input_file.replace(".csv", ".weighted.csv")
    print(f"Save summary to {save_file}")
    df.to_csv(save_file, index=False)


if __name__ == '__main__':
    args = get_options()

    if args.func == 'do_accu':
        do_accu(args)
    elif args.func == 'check_policy':
        check_policy(args)
    elif args.func == 'check_sft':
        check_sft(args)
    elif args.func == 'do_weighted':
        do_weighted(args)
    else:
        raise NotImplementedError
    

    

