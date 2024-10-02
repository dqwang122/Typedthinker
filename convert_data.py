import os
import json
import random

import argparse

from typing import List
from transformers import AutoTokenizer

random.seed(42)

DATA_ROOT_PATH = "datasets/"
MODEL_NAME_OR_PATH = "mistralai/Mistral-7B-Instruct-v0.2"

# python convert_data.py -i all.policy.jsonl -o all.policy -do_shuffle -do_split -input_key prompt -label_key target -type text2text -chat_format
# python convert_data.py -i all.sft.jsonl -o all.sft -do_shuffle -do_split -input_key prompt -label_key target -type text2text -chat_format


def get_options():
    args = argparse.ArgumentParser()
    # data options
    args.add_argument("-i", type=str, default="logic.policy.jsonl")
    args.add_argument("-o", type=str, default="logic.policy")
    args.add_argument("-input_key", type=str, default="input names, separated by comma")
    args.add_argument("-label_key", type=str, default="label names, separated by comma")
    args.add_argument("-type", type=str, default='text2text')
    args.add_argument("-do_shuffle", action='store_true', default=False)
    args.add_argument("-do_split", action='store_true', default=False)
    args.add_argument("-chat_format", action='store_true', default=False)

    args = args.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_options()
    print(args)

    input_file =  os.path.join(DATA_ROOT_PATH, args.i)
    output_file = os.path.join(DATA_ROOT_PATH, args.o)

    dataset = {
        "type": args.type, 
        "instances": []
    }

    if args.chat_format:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

    with open(input_file) as fin:
        for line in fin:
            data = json.loads(line)

            if args.chat_format:
                m = [{"role": "user", "content": data[args.input_key]}]
                prompt = tokenizer.apply_chat_template(m, tokenize=False)
            else:
                prompt = data[args.input_key]

            ex = {
                'id': data['id'],
                'benchmark': data['benchmark'],
                'task': data['task'],
                'input': prompt,
                'output': data[args.label_key]
            }
            dataset['instances'].append(ex)

    if args.do_shuffle:
        ids = list(set([d['id'] for d in dataset['instances']]))
        train_id = random.sample(ids, int(len(ids)*0.9))

    if args.do_split:
        train =  {
                    "type": args.type, 
                    "instances": [d for d in dataset['instances'] if d['id'] in train_id]
                }
        valid = {
                    "type": args.type, 
                    "instances": [d for d in dataset['instances'] if d['id'] not in train_id]
        }
        print("Convert {} examples to {}".format(len(train['instances']), output_file + ".train.json"))
        json.dump(train, open(output_file + ".train.json", 'w'))
        print("Convert {} examples to {}".format(len(valid['instances']), output_file + ".valid.json"))
        json.dump(valid, open(output_file + ".valid.json", 'w'))
    else:
        print("Convert {} examples to {}".format(len(dataset['instances']), output_file + ".json"))
        json.dump(dataset, open(output_file + ".json", 'w'))

            
      

