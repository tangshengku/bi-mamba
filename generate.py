# import debugpy
# debugpy.listen(("0.0.0.0", 8901))  # 将端口设置为8901
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()  # 让程序在此等待，直到调试器连接

import argparse
import torch
from tqdm import tqdm
from typing import Callable, Optional, Sequence, Union
from dataclasses import dataclass, field
from torch import Tensor
from model_utils.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer

from lm_eval.models.huggingface import HFLM
import lm_eval
import torch.nn as nn
from pathlib import Path
import json
from datautils import get_loaders
# from transformers import AutoTokenizer, AutoModelForCausalLM, MambaConfig, MambaForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
from utils import load_json, save_json
from qat.replace_module import replace_with_learnable_binarylinear
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer
import torch.nn.functional as F

import transformers
from utils import print_trainable_parameters

def load_open_src(model_name):
    model = MambaLMHeadModel.from_pretrained(model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", padding_side="left")
    return model, tokenizer

@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()



def load_ckpts(model_size, ckpt_dir, exist_extra_para, keep_parts, scaling_pattern):
    assert model_size in ["780M", "1.3B", "3B"]

    ckpt_dir = Path(ckpt_dir)
    if model_size == '3B':
        config = MambaConfig(
            d_model = 2560,
            vocab_size=32000,
            d_intermediate = 0,
            n_layer = 64,
            ssm_cfg={"layer": "Mamba2"},
            attn_cfg = {},
            attn_layer_idx = [],
            pad_vocab_size_multiple=16
            )
    elif model_size == '1.3B':
        config = MambaConfig(
            d_model = 2048,
            d_intermediate = 0,
            n_layer = 48,
            vocab_size=32000,
            ssm_cfg={"layer": "Mamba2"},
            attn_cfg = {},
            attn_layer_idx = [],
            pad_vocab_size_multiple=16,
            tie_embeddings = True
            )
    elif model_size == '780M':
        config = MambaConfig(
            d_model = 1536,
            d_intermediate = 0,
            n_layer = 48,
            vocab_size=32000,
            ssm_cfg={"layer": "Mamba2"},
            attn_cfg = {},
            attn_layer_idx = [],
            pad_vocab_size_multiple=16,
            tie_embeddings = True
            )
    elif model_size == '370M':
        config = MambaConfig(
            d_model = 1024,
            d_intermediate = 0,
            n_layer = 48,
            vocab_size=32000,
            ssm_cfg={"layer": "Mamba2"},
            attn_cfg = {},
            attn_layer_idx = [],
            pad_vocab_size_multiple=16,
            tie_embeddings = True
            )

    # model = MambaForCausalLM(config=config)
    print(model_size)
    print(config)
    model = MambaLMHeadModel(config).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b', padding_side="left", use_fast=False)

    if exist_extra_para:
        model = replace_with_learnable_binarylinear(model, scaling_pattern, keep_parts)

    weight_dict = {}
    ckpt_plist = [p for p in ckpt_dir.iterdir() if p.suffix == '.bin']
    for p in ckpt_plist:
        _weight_dict = torch.load(p)
        for k,v in _weight_dict.items():
            # if 'self_attn.rotary_emb.inv_freq' not in k:
            weight_dict[k] = v


    model.load_state_dict(weight_dict)
    for param in model.parameters():
        param.data = param.data.to(torch.float16)
        # param.data = param.data.to(torch.bfloat16)


    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens):
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.cuda()
    output_ids = model.generate(
        input_ids,
        max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
        )
    ouputs = []
    for ids in output_ids:
        output = tokenizer.decode(
                            ids,
                            spaces_between_special_tokens=False,
                            skip_special_tokens=False
                        )
        ouputs.append(output)
    return ouputs


def alpaca_generate(data_p, save_p, model, tokenizer, batch_size):
    with Path(data_p).open('r', encoding='utf-8') as f:
        data = json.load(f)
    result = {}
    batch = []
    save_id = 0
    save_p = Path(save_p)
    if save_p.exists():
        with save_p.open('r', encoding='utf-8') as r_f:
            result = json.load(r_f)
    for i, item in tqdm(enumerate(data)):
        if str(i) in result:
            save_id = i+1
            continue
        result[i] = item
        if item['input'] != '':
            prompt = item['instruction'] + ': ' + item["input"]
        else:
            prompt = item['instruction']

        prompt = prompt + '\nAnswer: '
        result[i]['prompt'] = prompt

        batch.append(prompt)
        if len(batch) == batch_size:
            outputs = generate(model, tokenizer, batch, 200)
            for ii, res in enumerate(outputs):
                result[save_id + ii]['res'] = res
                result[save_id + ii]['res_eos'] = res.split(tokenizer.eos_token)[0]
            batch = []
            save_id += batch_size
            with Path(save_p).open('w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)

    if len(batch) > 0:
        outputs = generate(model, tokenizer, batch, 200)
        for ii, res in enumerate(outputs):
            result[save_id + ii]['res'] = res
            result[save_id + ii]['res_eos'] = res.split(tokenizer.eos_token)[0]
        with Path(save_p).open('w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)


@torch.inference_mode()
def get_logits(
    input_ids,
    model,
    max_length,
    teacher_outputs=None,
    vocab_size=None,
    cg=False,
    streamer: Optional[TextStreamer] = None
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    if streamer is not None:
        streamer.put(input_ids.cpu())

    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

    def _get_logits(input_ids, inference_params):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            position_ids = torch.full(
                (batch_size, 1),
                inference_params.seqlen_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
            #print('position_ids', position_ids)
        else:
            position_ids = None
            #print('position_ids', position_ids)

        if not cg or not decoding:
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits.squeeze(dim=1)
            # print('logits', logits.shape)

        else:
            logits = model._decoding_cache.run(
                input_ids, position_ids, inference_params.seqlen_offset
            ).squeeze(dim=1)
            #print('logits', logits)
        return logits[..., :vocab_size] if vocab_size is not None else logits

    logits = _get_logits(input_ids, inference_params)

    return logits
        


def check_logit(model, tokenizer, prompt):
    input_ids = tokenizer([prompt], return_tensors = 'pt').input_ids
    print(input_ids.shape)
    for i in range(len(input_ids[0])):
        logits = get_logits(input_ids[:,:i+1].to('cuda'), model, max_length=64)
        top5_token = logits[0].topk(20).indices
        label = input_ids[0,i+1]
        print(tokenizer.decode(top5_token), '|', tokenizer.decode([label]))

def check_groundtruth(model, tokenizer, batch_size=16):
    _, testdata = get_loaders('wikitext2', tokenizer)
    test_tkids = testdata.input_ids[0][:batch_size*2048].reshape(batch_size, 2048)
    result = {
        'prob': {k:[] for k in ['right', 'cur'] + list(range(1, 6))},
        'match_right_rate': {k:0 for k in range(1, 6)},
        'match_cur_rate': {k:0 for k in range(1, 6)},
    }
    all_num = 0
    for i in tqdm(range(2047)):
        logits = get_logits(test_tkids[:,:i+1].to('cuda'), model, max_length=64) # (batch_size, vocab_size)
        probs = torch.softmax(logits, -1) # (batch_size, vocab_size)

        label_id = test_tkids[:,i+1].to('cuda') # (batch_size)
        label_sel = torch.nn.functional.one_hot(label_id, num_classes=logits.shape[-1]).to(torch.bool)
        right_prob = probs[label_sel]
        result['prob']['right'].append(right_prob)

        cur_id = test_tkids[:,i].to('cuda') # (batch_size)
        cur_sel = torch.nn.functional.one_hot(cur_id, num_classes=logits.shape[-1]).to(torch.bool)
        cur_prob = probs[cur_sel]
        result['prob']['cur'].append(cur_prob)

        top_res = probs.topk(5)
        all_num += batch_size
        for k in range(5):
            result['prob'][k+1].append(top_res.values[:,k])

            _match_right_num = (top_res.indices[:, k] == label_id).sum()
            result['match_right_rate'][k+1] += _match_right_num

            _match_cur_num = (top_res.indices[:, k] == cur_id).sum()
            result['match_cur_rate'][k+1] += _match_cur_num

    for k in result['prob']:
        result['prob'][k] = float(torch.concat(result['prob'][k]).mean())
    for k in result['match_right_rate']:
        result['match_right_rate'][k] = float(result['match_right_rate'][k]/all_num)
    for k in result['match_cur_rate']:
        result['match_cur_rate'][k] = float(result['match_cur_rate'][k]/all_num)

    print(result)
    return result


# def generate(model, tokenizer, prompt, max_length):
#     input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).cuda()
#     res = []
#     sample_tkid = None
#     while len(res) < max_length and sample_tkid != tokenizer.eos_token_id:
#         logits = model(input_ids).logits
#         sel_ids = torch.argmax(logits, dim=-1)
#         sample_tkid = sel_ids[0, -1]
#         res.append(sample_tkid)
#         input_ids = torch.cat([input_ids, sel_ids[:, -1].unsqueeze(-1)], -1)
    
#     tk = tokenizer.decode(res)
#     return tk


def main(args):
    result = {}
    for ckpt_id in args.ckpt_ids.split(','):
        ckpt_dir = Path(args.ckpt_dir)
        model, tokenizer = load_ckpts(
            args.model_size, 
            ckpt_dir/f'ckpt-{ckpt_id}',  
            args.exist_extra_para, 
            args.keep_parts, 
            args.scaling_pattern
            )
        res = check_groundtruth(model, tokenizer)
        result[f'ckpt-{ckpt_id}'] = res
        with Path(f'eval_result/{ckpt_dir.stem}_{"-".join(args.ckpt_ids.split(","))}.json').open('w') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='3B')
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--exist_extra_para', action="store_true")
    parser.add_argument("--eval_open_src", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not args.eval_open_src:
        model, tokenizer = load_ckpts(
            args.model_size, 
            model_dir,  
            True, 
            ["lm_head"], 
            'column'
            )
    else:
        model, tokenizer = load_open_src(model_dir)

    save_dir = Path('generate_res')
    save_dir.mkdir(exist_ok=True)
    save_path = f'generate_res/single_{model_dir.parent.name}_{model_dir.name}.json'
    alpaca_generate("data/alpaca_data.json", save_path, model, tokenizer, 1)

    # prompts = [
    #     "Explain the use of word embeddings in Natural Language Processing\nAnswer: "
    # ]
    # res = generate(model, tokenizer, prompts, 200)
    # for r in res:
    #     print(r)
    #     print('\n')
    # print('='*32)