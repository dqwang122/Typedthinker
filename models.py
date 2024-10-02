import os
import openai
import backoff 

from tqdm import tqdm
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from functools import partial



MAX_LENGTH=4096

MODEL_NAME_PATH = {
    "llama": "yahma/llama-7b-hf",
    "llama2": "meta-llama/Llama-2-7b-hf",
    "llama2_chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama3_chat": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mistral_chat": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral_moe": "mistralai/Mixtral-8x7B-v0.1",
    "mistral_moe_chat": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}


class HuggingfaceLLM():
    def __init__(self, model_name: str = "mistral", padding_side: str = "left"):
        super().__init__()

        self.model_name = model_name.split('/')[-1]
        
        model_name_or_path = MODEL_NAME_PATH.get(model_name, model_name)
        self.model_name_or_path = model_name_or_path

        print(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side=padding_side,
            use_fast=False
        )
        self.tokenizer.padding_side = padding_side
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("Vocab Size: ", len(self.tokenizer))
        print("Loaded in model and tokenizer! Padding side:", padding_side)

    def get_chat_message(self, prompts, sys):
        messages = []

        for p in prompts:                
            m = [{"role": "user", "content": p}]
            messages.append(m)

        if sys is not None and sys != '':
            for m in messages:
                if self.model_name.startswith('mistral'):
                    m[0] = {"role": "user", "content": sys + '\n\n' + m[0]['content']}
                elif self.model_name.startswith('llama'):
                    m.insert(0, {"role": "system", "content": sys})

        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False) for m in messages]
        return prompts

    def inference(self, prompts: List, batch_size=1, max_tokens=1000, n=1, stop=None, sys=None, verbose=True, chat_format=True, **kwargs) -> Iterable[str]:
        self.model.eval()
        
        if not isinstance(prompts, list):
            prompts = [prompts]

        example_num = len(prompts)
        if example_num == 0:
            return []

        if n > 1:
            prompts = [x for x in prompts for _ in range(n)]

        if chat_format:
            prompts = self.get_chat_message(prompts, sys)

        results = []
        chunk_num = int(len(prompts) // batch_size) + (len(prompts) % batch_size > 0)
        scope = tqdm(range(chunk_num)) if chunk_num > 1 else range(chunk_num)
        
        kwargs['max_new_tokens'] = max_tokens
        kwargs['do_sample'] = True if kwargs.get('temperature', 1.0) > 0 else False

        if verbose:
            print('='*50 + 'Prompt' + '='*50)
            print('prompts', len(prompts))
            print(prompts[0].replace('\n',' '))

        with torch.no_grad():
            for i in scope:
                batch_x = prompts[i*batch_size:(i+1)*batch_size]
                inputs = self.tokenizer(batch_x, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(self.model.device)
                prompt_len = inputs.input_ids.shape[1]
                outputs = self.model.generate(**inputs, **kwargs)
                outputs = outputs[:,prompt_len:]
                res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend(res)

                if verbose and i == 0:
                    print('='*50 + 'results' + '='*50)
                    print('results', len(results))
                    print(results[0])
                    print('='* 100)

        if n > 1:
            results = [results[i:i+n] for i in range(0, len(results), n)]
        return results


class HuggingfacevLLM():
    def __init__(self, model_name: str = "mistral", padding_side: str = "left", device_num: int=1, max_tokens=4096):
        super().__init__()

        self.model_name = model_name.split('/')[-1]
        self.model_name_or_path = MODEL_NAME_PATH.get(model_name, model_name)
        self.device_num = device_num

        print(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side=padding_side,
            use_fast=False
        )
        self.tokenizer.padding_side = padding_side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = LLM(
                model=self.model_name_or_path,
                tokenizer=self.model_name_or_path,
                tensor_parallel_size=self.device_num,
                dtype="bfloat16",
                # enforce_eager=True,
                max_model_len=max_tokens,
                # disable_custom_all_reduce=False,
                # enable_prefix_caching=True,
            )

        print("Vocab Size: ", len(self.tokenizer))
        print("Loaded in model and tokenizer! Padding side:", padding_side)

    def get_chat_message(self, prompts, sys):
        messages = []

        for p in prompts:                
            m = [{"role": "user", "content": p}]
            messages.append(m)

        if sys is not None and sys != '':
            for m in messages:
                if self.model_name.startswith('mistral'):
                    m[0] = {"role": "user", "content": sys + '\n\n' + m[0]['content']}
                elif self.model_name.startswith('llama'):
                    m.insert(0, {"role": "system", "content": sys})

        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False) for m in messages]
        return prompts

    def inference(self, prompts: List, batch_size=1, max_tokens=1000, n=1, stop=None, sys=None, verbose=True, chat_format=True, **kwargs) -> Iterable[str]:
        if not isinstance(prompts, list):
            prompts = [prompts]
        if len(prompts) == 0:
            return []
        if chat_format:
            prompts = self.get_chat_message(prompts, sys)

        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=kwargs.get('temperature', 0.0),
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop,
            skip_special_tokens=True, 
        )

        vllm_outputs = self.model.generate(prompts, sampling_params)
        results = [[o.text for o in vllm_output.outputs] for vllm_output in vllm_outputs]

        if verbose:
            print('='*50 + 'Prompt' + '='*50)
            print('prompts', len(prompts))
            print(prompts[0].replace('\n',' '))
            print('='*50 + 'results' + '='*50)
            print('results', len(results))
            print(results[0])
            print('='* 100)

        if n == 1:
            results = [r[0] for r in results]
        return results
    
def load_model(args, model_name, max_tokens, device, sys=''):
    llm = HuggingfaceLLM(model_name=model_name, device=device)
    llmfunc = partial(llm.inference, batch_size=args.batch_size, temperature=args.temperature, max_tokens=max_tokens, sys=sys)
    print("[INFO] Load Model: ", llmfunc)
    return llmfunc