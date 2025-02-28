from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
import json
import time
import multiprocessing
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from torch.utils.data import Dataset
import transformers
from typing import Dict, Optional, Sequence


def tokenize_single(data_p_list, tokenizer, output_dir):
    print(len(data_p_list))
    data_p_list = sorted(data_p_list, key=lambda x: int(x.stem.split("_")[-1]))
    for data_p in tqdm(data_p_list):
        text, meta = read_slimpajama_jsonl(data_p)
        input_ids = tokenizer(text)['input_ids']
        data = []
        for ids, mt in zip(input_ids, meta):
            data.append(json.dumps({'tk_ids': ids, 'meta': mt}))
            
        with (output_dir / f'{data_p.stem.split("_")[-1]}.jsonl').open('w', encoding='utf-8') as w_f:
            w_f.write('\n'.join(data))

def tokenize_multi(data_dir, output_dir, divide_num = 4):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print('pad_token', tokenizer.pad_token, tokenizer.pad_token_id)
    print('bos_token', tokenizer.bos_token, tokenizer.bos_token_id)
    print('eos_token', tokenizer.eos_token, tokenizer.eos_token_id)
    print('unk_token', tokenizer.unk_token, tokenizer.unk_token_id)

    data_p_list = list(data_dir.iterdir())
    data_p_list = sorted(data_p_list, key=lambda x: int(x.stem.split("_")[-1]))

    step = (len(data_p_list) + divide_num - 1 ) // divide_num
    data_p_list_group = [data_p_list[i * step: (i + 1) * step] for i in range(divide_num)]

    func = partial(tokenize_single, tokenizer = tokenizer, output_dir = output_dir)
    with Pool(divide_num) as pool:
        pool.map(func, data_p_list_group)
            
def tokenize_omission(data_dir, output_dir):
    for chunk_id in range(1, 11):
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')

        chunk_dir = data_dir / f'chunk{chunk_id}'
        output_chunk_dir = output_dir / f'chunk{chunk_id}'
        if not output_chunk_dir.exists():
            output_chunk_dir.mkdir(exist_ok=True, parents=True)
        processed_id_list  = [p.stem for p in output_chunk_dir.iterdir()]
        print(f'chunk {chunk_id} processed file num: ', len(processed_id_list))
        print(f'chunk {chunk_id} all file num: ', len([p for p in chunk_dir.iterdir()]))
        data_p_list = [p for p in chunk_dir.iterdir() if p.stem.split("_")[-1] not in processed_id_list]
        print(f'chunk {chunk_id} unprocessed file num: ', len(data_p_list))
        data_p_list = sorted(data_p_list, key=lambda x: int(x.stem.split("_")[-1]))
        for data_p in tqdm(data_p_list):
            text, meta = read_slimpajama_jsonl(data_p)
            input_ids = tokenizer(text)['input_ids']
            data = []
            for ids, mt in zip(input_ids, meta):
                data.append(json.dumps({'tk_ids': ids, 'meta': mt}))
                
            with (output_dir / f'chunk{chunk_id}' / f'{data_p.stem.split("_")[-1]}.jsonl').open('w', encoding='utf-8') as w_f:
                w_f.write('\n'.join(data))

def split_data(tokenizer, data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    buffer = []
    new_data = []
    chunk_num = 0
    for data_p in tqdm(data_dir.iterdir()):
        with data_p.open('r', encoding='utf-8') as r_f:
            for line in r_f:
                _item = json.loads(line)
                buffer.extend(_item['tk_ids'] + [tokenizer.eos_token_id])
                while len(buffer) >= 2049:
                    new_data.append(json.dumps({"token_ids": buffer[:2049]}))
                    if buffer[-1] != tokenizer.eos_token_id:
                        buffer = buffer[2048:]
                    else:
                        buffer = buffer[2049:]
                    if len(new_data) == 1706976:
                        with (output_dir / f'{chunk_num}.jsonl').open('w', encoding='utf-8') as w_f:
                            w_f.write('\n'.join(new_data))
                        new_data = []
                        chunk_num += 1
    if len(buffer) > 0:
        new_data.append(json.dumps({"token_ids": buffer}))
    if len(new_data) > 0:
        with (output_dir / f'{chunk_num}.jsonl').open('w', encoding='utf-8') as w_f:
            w_f.write('\n'.join(new_data))


def read_slimpajama_jsonl(data_path):   
    text, meta = [], []
    with data_path.open('r', encoding='utf-8') as r_f:
        for line in r_f:
            json_data = json.loads(line.strip())
            text.append(json_data["text"])
            meta.append(json_data["meta"])
    return text, meta
        


if __name__ == '__main__':
    # tokenize_multi(data_dir = "data/chunk1", output_dir = "data/chunk1_tokenized")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    split_data(tokenizer, "data/chunk1_tokenized", "data/Slimpajama")