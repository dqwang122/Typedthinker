import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from typing import Dict, Sequence
from dataclasses import dataclass

import copy

import transformers
from transformers import EvalPrediction

import evaluate
import numpy as np

IGNORE_INDEX = -100
max_length = 1024

class SupervisedLLMDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, raw_dataset: Sequence[Dict]):
        super().__init__()
        self.tokenizer = tokenizer
        targets = [
            f"{example['output']}{self.tokenizer.eos_token}" for example in raw_dataset
        ]
        data_dict = self._preprocess(raw_dataset["input"], targets)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        # print(len(self.input_ids), self.input_ids[0].size(), self.labels[0].size())

    def _preprocess(self, sources, targets):
        # remove pairs where at least one record is None
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            _tokenize_fn(strings, self.tokenizer) for strings in (examples, sources)
        ]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class SupervisedCLSDataset(SupervisedLLMDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, raw_dataset: Sequence[Dict]):
        super(Dataset, self).__init__()
        self.tokenizer = tokenizer
        targets = [
            example['output'] for example in raw_dataset
        ]
        data_dict = self._preprocess(raw_dataset["input"], targets)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        # print(len(self.input_ids), self.input_ids[0].size(), self.labels[0].size())

    def _preprocess(self, sources, targets):
        examples_tokenized = _tokenize_fn(sources, self.tokenizer)
        input_ids = examples_tokenized["input_ids"]
        targets = [torch.tensor([x]) for x in targets]
        return dict(input_ids=input_ids, labels=targets)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], tag_labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        keys = instances[0].keys()
        batch_data = {}
        for key in keys:
            if key not in ["input_ids", "token_labels", "tag_labels", "value_labels", "labels"]:
                raise ValueError(f"Invalid key: {key}")
            if key == "input_ids":
                input_ids = [instance[key] for instance in instances]
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
                )
                batch_data['input_ids'] = input_ids
                batch_data['attention_mask'] = input_ids.ne(self.tokenizer.pad_token_id)
            else:
                labels = [instance[key] for instance in instances]
                batch_data[key] = torch.nn.utils.rnn.pad_sequence(
                    labels, batch_first=True, padding_value=IGNORE_INDEX
                )
        
        # print(batch_data)
        # print(batch_data['input_ids'].shape)
        return batch_data

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, train: Sequence[Dict], valid: Sequence[Dict] = None, mode="text2text") -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    is_regression = False
    if mode == 'text2text':
        DATASETCLASS = SupervisedLLMDataset
    elif mode == 'text2tag':
        DATASETCLASS = SupervisedCLSDataset
    elif mode == 'text2value':
        is_regression = True
        DATASETCLASS = SupervisedCLSDataset
    else:
        raise ValueError(f"Invalid mode: {mode}")

    print('[INFO] Processing the training dataset... ')
    train_dataset = DATASETCLASS(tokenizer=tokenizer, raw_dataset = train)
    if valid:
        print('[INFO] Processing the valid dataset...')
        eval_dataset = DATASETCLASS(tokenizer=tokenizer, raw_dataset = valid)
    else:
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print('[INFO] Finish Dataset and Collator creation.')

    metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    
    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, compute_metrics=compute_metrics
    )







def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length = max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )