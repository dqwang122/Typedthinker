import os, sys
import copy
import json
import click
import datetime
import argparse

import yaml

import numpy as np

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, AdamW, Trainer

from dataloader import make_supervised_data_module
from agent import ReasoningAgent




os.environ["WANDB_PROJECT"] = "intrilora"  # name your W&B project
# os.environ["WANDB_LOG_MODEL"] = "false"  # log all model checkpoints

KEY_TYPE = "type"
KEY_INSTANCES = "instances"
ds_config = "configs/ds_config_zero3.json"


# deepspeed --num_gpus 4 finetune.py
# deepspeed --master_port 61000 --include "localhost:3,4" finetune.py --data_file datasets/logic.policy.json --save_dir checkpoints/
# CUDA_VISIBLE_DEVICES=7 python finetune.py --data_file dataset/logic.policy.train.json --save_dir checkpoints/

def get_options():
    parser = argparse.ArgumentParser()
    # data options
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_file", type=str, default="dataset/sft/logic.policy.train.json")
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints/")
    parser.add_argument("--backend", type=str, default="mistral_chat")
    parser.add_argument("--padding_strategy", type=str, default='left')
    parser.add_argument("--do_eval", action='store_true', default=False)
    parser.add_argument("--from_checkpoint", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="text2text", choices=["text2tag", "text2text"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument('--max_steps', type=int, default=4000)    # (example_size * num_epochs) / (batch_size * gpu)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--deepspeed', action='store_true', default=False)
    parser.add_argument('--lora', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_name', type=str, default="intrilora")

    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        arg_dict = vars(args)
        for key, value in config.items():
            # Only set the attribute if it's not already set by a command-line argument 
            # which means that it is the same with the default value
            if arg_dict[key] == parser.get_default(key):
                setattr(args, key, value)

    args.do_train = False if args.do_eval else True

    return args


if __name__ == '__main__':

    # Note args contains the values with the priority:
    # command-line value > config file value > default
    args = get_options()
    print(args)

    input_file =  args.data_file
    output_dir = args.save_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    yaml.dump(vars(args), open(os.path.join(output_dir, "args.yaml"), "w"))

    # sanity check over the fields of json file
    with open(input_file) as fin:
        json_data = json.load(fin)
        if KEY_TYPE not in json_data.keys():
            raise ValueError(
                f'"{KEY_TYPE}" field must be specified for data, e.g.'
                "{\n"
                f'   "{KEY_TYPE}: "text2text",\n'
                f'   "{KEY_INSTANCES}": [\n'
                '       { "text": "Sentence 1: This is a sentence." }\n'
                '       { "text": "Sentence 2: This is another sentence." }\n'
                f"   ]\n"
                "}"
            )

    # Load the dataset using the HuggingFace dataset library
    extensions = "json"
    raw_dataset = load_dataset(
        extensions,
        data_files=[input_file],
        field=KEY_INSTANCES,
        split="train",
        token=None,
    )

    print(raw_dataset)
    print(raw_dataset[1])

    if args.valid_file:
        valid_dataset = load_dataset(
            extensions,
            data_files=[args.valid_file],
            field=KEY_INSTANCES,
            split="train",
            token=None,
        )
    else:
        valid_dataset = None

    
    if args.mode == "text2text":
        label_names = ["labels"]
    else:
        raise ValueError(f"Unknown mode {args.mode}")
        
    model, tokenizer = ReasoningAgent(model_name=args.backend).prepare_for_finetune(args.lora)

    run_name = f"{args.mode}_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
    data_module = make_supervised_data_module(tokenizer=tokenizer, train=raw_dataset, valid=valid_dataset, mode=args.mode)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps" if args.valid_file else "no",
        eval_steps = args.save_steps,
        label_names = label_names,
        log_level="info",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0,
        weight_decay=0,
        # max_steps=args.max_steps,
        num_train_epochs = args.max_epochs,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True if args.valid_file else False,
        metric_for_best_model='loss' if args.mode == "text2text" else 'accuracy',
        greater_is_better=False if args.mode == "text2text" else True,
        seed=args.seed,
        run_name=run_name,
        deepspeed=ds_config if args.deepspeed else None,
        log_on_each_node=False,
        fp16=False,
        bf16=True,
        tf32=False,
        report_to='wandb' if args.wandb else "none",
    )  # tf32=True -> only for A100


    ignore_keys_for_eval=None

    if args.do_train:
        print("[INFO] Start the trainer")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data_module["train_dataset"],
            eval_dataset=data_module["eval_dataset"],
            tokenizer=tokenizer,
            data_collator=data_module["data_collator"],
            compute_metrics=data_module["compute_metrics"],
            optimizers=(None, None),
            preprocess_logits_for_metrics=None,
        )


        train_result = trainer.train(ignore_keys_for_eval=ignore_keys_for_eval)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        if args.valid_file:
            eval_result = trainer.evaluate(ignore_keys=ignore_keys_for_eval)
            # metrics = eval_result['metrics']
            print(eval_result)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        trainer.save_state()
        
    else:
        print("[INFO] Start the predictor")
        predict_args = TrainingArguments(
                            output_dir=output_dir,
                            label_names = label_names,
                            log_level="info",
                            per_device_eval_batch_size=args.batch_size,
                            eval_accumulation_steps=4,
                            seed=args.seed,
                            metric_for_best_model='loss' if args.mode == "text2text" else 'accuracy',
                            greater_is_better=False if args.mode == "text2text" else True,
                            fp16=False,
                            bf16=True,
                            tf32=False,)

        trainer = Trainer(
            model=model,
            args=predict_args,
            tokenizer=tokenizer,
            data_collator=data_module["data_collator"],
            compute_metrics=data_module["compute_metrics"],
        )

        results = trainer.predict(data_module["eval_dataset"], ignore_keys=ignore_keys_for_eval)
        prediction = results.predictions.tolist()
        labels = results.label_ids.tolist()
        metrics = results.metrics
        result_dict = {
            "prediction": prediction,
            "labels": labels,
            "metrics": metrics,
        }
        json.dump(result_dict, open(os.path.join(output_dir, "results.json"), "w"))
        