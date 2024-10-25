from tqdm import tqdm
from pathlib import Path
import json
import time
import multiprocessing
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import transformers
from typing import Dict, Optional, Sequence

import torch
import copy

from dataclasses import dataclass, field
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM

import numpy as np
import wandb
import argparse

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=[inp.numpy().tolist() for inp in input_ids], labels=[lab.numpy().tolist() for lab in labels])


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        processed_p = Path('data/tokenized_alpacas.json')
        if not processed_p.exists():
            super(SupervisedDataset, self).__init__()
            print("Loading data...")
            with Path(data_path).open('r', encoding='utf-8') as r_f:
                list_data_dict = json.load(r_f)

            print("Formatting inputs...")
            prompt_input, prompt_no_input = ALPACA_PROMPT_DICT["prompt_input"], ALPACA_PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in list_data_dict
            ]
            targets = [f"{example['output']} {tokenizer.eos_token}" for example in list_data_dict]

            print("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)

            data = []
            for inp, lab in zip(data_dict['input_ids'], data_dict['labels']):
                data.append(json.dumps(dict(input_ids=inp, labels=lab)))
            
            with processed_p.open('w', encoding='utf-8') as w_f:
                w_f.write('\n'.join(data))
        else:
            with processed_p.open('r', encoding='utf-8') as r_f:
                data = json.load(r_f)

        self.input_ids = [torch.tensor(json.loads(line)['input_ids']) for line in data]
        self.labels = [torch.tensor(json.loads(line)['labels']) for line in data]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

if __name__ == '__main__':
    get_alpaca_docs()
    tokenize_alpaca_docs()