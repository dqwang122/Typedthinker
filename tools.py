import os
import sys
import re
import yaml
import json
import math

import pandas as pd

from retrieval import Memory

def last_boxed_only_string(string):
        idx = string.rfind("oxed") #change
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval

def remove_boxed(s):
    left = "oxed{" #change
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        answer = s[len(left):-1]
        if "=" in answer:
            answer = answer.split("=")[-1].lstrip(" ")
        return answer
    except:
        return None


def most_frequent(value, weight=1):
    if weight == 1:
        value = [v for v in value if v is not None]
    else:
        value = [[v] * int(w*10) for v,w in zip(value,weight) if v is not None]
        value = sum(value, [])
    if not value:
        return None
    return max(sorted(set(value)), key = value.count)


def get_box_option(x, strict=False):
    content = remove_boxed(last_boxed_only_string(x))
    option_pattern = r"(\([A-Za-z]\))"
    if content:
        match = re.findall(option_pattern, content)
        if match:
            if len(match) > 1:  # multiple options
                return None
            return match[0].upper()
        if not strict:
            option_pattern = r"([A-Za-z])"
            match = re.findall(option_pattern, content)
            if match:
                if len(match) > 1:  # multiple options
                    return None
                return '(' + match[0].upper() + ')'
    return None

def get_plain_option(content):
    result = ''
    pattern = r".*(\([A-Za-z]\))"
    if re.findall(pattern, content) != []:
        result = re.findall(pattern, content)[0]
    else:
        pattern = r"[Oo]ption ([A-Za-z])[\W]|[Tt]he answer is ([A-Za-z])|Answer: ([A-Za-z])|(\([A-Za-z]\))|\[([A-Za-z])\]"
        m = re.findall(pattern, content)
        if m != []:
            # result = [x for x in m[-1] if x != ''][0]
            for i in range(5):
                cand = [x[i] for x in m if x[i] != '']
                if len(cand) == 0:  # no option found
                    continue
                elif len(cand) > 1:  # multiple options found
                    return None
                else:
                    result = cand[0]
                    break
            if result[0] != '(':
                result = '(' + result
            if result[-1] != ')':
                result = result + ')'
        else:
            if '.' in content and content.split('.')[0].strip().isupper():
                result = '(' + content.split('.')[0].strip() + ')'
            elif ':' in content and content.split(':')[0].strip().isupper():
                result = '(' + content.split(':')[0].strip() + ')'
            else:
                # print(content)
                pass
    if result == '':
        return None
    return result.upper()


def get_mixed_option(content, strict=False):
    result = get_box_option(content, strict=strict)
    if result is None:
        result = get_plain_option(content)
    return result


def get_policy(content):
    content = content.replace('<|start_header_id|>', '')
    res = {}
    if not content.startswith('['):
        content = '[' + content
    left = content.find('[')
    right = content.rfind(']')
    key_content = content[left:right+1]
    try:
        key_content = eval(key_content)
    except:
        print(content, key_content)
        return {'deductive': -1, 'inductive': -1, 'analogical': -1, 'abductive': -1, 'none': 1}
    for x in key_content:
        if 'Confidence' not in x or 'ReasoningType' not in x:
            x['Confidence'] = -1
        if not isinstance(x['ReasoningType'], str):
            continue
        res[x['ReasoningType'].lower()] = x['Confidence']
    for key in EXAMPLE_TYPE:
        if key not in res:
            res[key] = -1
    return res
    

from scipy.stats import spearmanr, pearsonr, kendalltau
import numpy as np

def calculate_correlation(pred_score, human_score):
    result = {'pearsonr': 0, 'spearmanr': 0, 'kendalltau': 0}
    assert len(pred_score) == len(human_score)
    print(len(pred_score))

    pearsonr_res = pearsonr(pred_score, human_score)
    spearmanr_res = spearmanr(pred_score, human_score)
    kendalltau_res = kendalltau(pred_score, human_score)
    result['pearsonr'], result['pearsonr_status'] = pearsonr_res[0], pearsonr_res[1:]
    result['spearmanr'], result['spearmanr_status'] = spearmanr_res[0], spearmanr_res[1:]
    result['kendalltau'], result['kendalltau_status'] = kendalltau_res[0], kendalltau_res[1:]

    match = (pred_score == human_score).sum()
    accu = match / len(pred_score)
    result['accu'] = accu

    return result

def compute_metric(expected, labels):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    return {
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }


sys.path.append("clone/NeMo-Skills")
from nemo_skills.inference.prompt.utils import prompt_types, Prompt, PromptConfig, FewShotExamples
PROMPT_YAML_PATH = "clone/NeMo-Skills/nemo_skills/inference/prompt/"

DEFINITIONS = {
    "deductive": "to deduce conclusion based on the general rules and premise.",
    "inductive": "to make broad generalizations from specific observations.",
    "analogical": "to retrieve several relevant information and draw the conclusion of this problem based on the similarity.",
    "abductive": "to assume one candidate is correct and check whether it meets the condition in the problem.",
    "text": "",
}

TEMPLATE = """Use {reasoning} reasoning to solve the given question. {reasoning} is {definition}.\n\nQuestion: {question}\n\nMy solution:
"""

EXAMPLE_TYPE = ['deductive', 'inductive', 'analogical', 'abductive', 'text']


def make_prompt(data, problem, mode, examples_type, prompt_type="base"):
    file = f"{PROMPT_YAML_PATH}/{prompt_type}.yaml"
    prompt_config_file = yaml.safe_load(open(file))
    few_show_config = prompt_config_file['few_shot_examples']

    if examples_type != 'all':
        examples_type_list = [examples_type] if isinstance(examples_type, str) else examples_type
    else:
        examples_type_list = EXAMPLE_TYPE


    if mode == 'none':
        for d in data:
            d['prompt'] = d['question']
        return data
    elif mode == 'definition':
        new_data = []
        for d in data:
            for etype in examples_type_list:
                ins = copy.deepcopy(d)
                ins['prompt_type'] = etype
                if etype == 'text':
                    ins['prompt'] = ins['question']
                else:
                    ins['prompt'] = TEMPLATE.format(reasoning=etype.capitalize(), definition=DEFINITIONS[etype], question=ins['question'])
                new_data.append(ins)
        return new_data
    else:
        new_data = []
        prompt_config_list = []
        for examples_type in examples_type_list:
            prompt_config_file['few_shot_examples'] = FewShotExamples(
                                                        examples_type=problem + '_' + examples_type, 
                                                        num_few_shots=3, 
                                                        **few_show_config)
            prompt_config = PromptConfig(**prompt_config_file)
            prompt_config_list.append(prompt_config)


        for d in data:
            for prompt_config in prompt_config_list:
                d = d.copy()
                d['prompt_type'] = prompt_config.few_shot_examples.examples_type
                if d['prompt_type'] == 'math_analogical':
                    p = Prompt(prompt_config, d, example_dicts=None, context_template="Use analogical reasoning based on a similar question and its solution:\n{retrieval}\n\n")
                else:
                    p = Prompt(prompt_config, d, example_dicts=None)
                d['prompt'] = str(p)
                new_data.append(d) 
    
    return new_data


CORPUS_ROOT_PATH = "datasets/store"
def make_rag_prompt(data, benchmark, problem, examples_type, prompt_type="base"):
    file = f"{PROMPT_YAML_PATH}/{prompt_type}.yaml"
    prompt_config_file = yaml.safe_load(open(file))
    few_show_config = prompt_config_file['few_shot_examples']
    if examples_type != 'all':
        examples_type_list = [examples_type] if isinstance(examples_type, str) else examples_type
    else:
        examples_type_list = EXAMPLE_TYPE

    new_data = []
    for examples_type in examples_type_list:
        prompt_config_file['few_shot_examples'] = FewShotExamples(
                                                    examples_type=problem + '_' + examples_type, 
                                                    num_few_shots=3, 
                                                    **few_show_config)
        prompt_config = PromptConfig(**prompt_config_file)

        # load memory
        if examples_type == 'none' or examples_type == 'text':
            corpus_file = f"{CORPUS_ROOT_PATH}/{benchmark}.raw.jsonl"
        else:
            corpus_file = f"{CORPUS_ROOT_PATH}/{benchmark}.{examples_type}.jsonl"
        df = pd.read_json(corpus_file, orient='records', lines=True)
        print(f"Load {df.shape[0]} memory from {corpus_file}...")
        corpus = list(zip(df['question'].values,df['solution'].values))
        print(corpus[0])
        explicit_mem = Memory(corpus, memory_size=30000, device='cuda')

        queries = [d['question'] for d in data]
        retrieved = explicit_mem.retrieve(queries, top_k=3, thre=0.5, batch_size=10000, verbose=False)

        for d, r in zip(data, retrieved):
            d = d.copy()
            query = d['question']
            if r != []:
                example_dicts = [{'question': x[0], 'generated_solution': x[1]} for x in r]   
            else:
                example_dicts = None

            d['prompt_type'] = prompt_config.few_shot_examples.examples_type
            p = Prompt(prompt_config, d, example_dicts=example_dicts)
            
            d['prompt'] = str(p)
            new_data.append(d) 
    
    return new_data