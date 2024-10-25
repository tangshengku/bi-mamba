from transformers import AutoTokenizer
from transformers import AutoTokenizer
# from mamba_ssm.models.config_mamba import MambaConfig
import torch
import json
from pathlib import Path
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from qat.replace_module import replace_with_learnable_binarylinear

def load_ckpts(model_size, ckpt_dir, ckpt_type, exist_extra_para, keep_parts, scaling_pattern):
    assert model_size in ["780M", "1.3B", "3B"]
    assert ckpt_type in ['torch', 'hf_st', 'lightning']

    ckpt_dir = Path(ckpt_dir)
    if model_size == '3B':
        # with Path(f'mamba2_{model_size}.json').open('r') as r_f:
        #     _config = json.load(r_f)
        # config = MambaConfig(**_config)
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

   
    print(model_size)
    print(config)
    model = MambaLMHeadModel(config)
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b', padding_side="right", use_fast=False)

    if exist_extra_para:
        model = replace_with_learnable_binarylinear(model, scaling_pattern, keep_parts)
        key_list = []
        for k, v in model.named_parameters():
            key_list.append(k)
        with Path('para_key_bimamba_3b_before.txt').open('w') as w_f:
            w_f.write('\n'.join(key_list))

    weight_dict = {}
    if ckpt_type == 'torch':
        ckpt_plist = [p for p in ckpt_dir.iterdir() if p.suffix == '.bin']
        for p in ckpt_plist:
            _weight_dict = torch.load(p)
            for k,v in _weight_dict.items():
                # if 'self_attn.rotary_emb.inv_freq' not in k:
                weight_dict[k] = v
    
    key_list = []
    for k in weight_dict:
        key_list.append(k)
    with Path('para_key_bimamba_3b.txt').open('w') as w_f:
        w_f.write('\n'.join(key_list))


    model.load_state_dict(weight_dict)
    for param in model.parameters():
        param.data = param.data.to(torch.float16)
        # param.data = param.data.to(torch.bfloat16)

    print(model.backbone.embedding.weight)
    print(model.lm_head.weight)


    return model, tokenizer

def load_open_src(model_name):
    weight_dict = {}
    ckpt_plist = [p for p in Path(model_name).iterdir() if p.suffix == '.bin']
    for p in ckpt_plist:
        _weight_dict = torch.load(p)
        for k,v in _weight_dict.items():
            # if 'self_attn.rotary_emb.inv_freq' not in k:
            weight_dict[k] = v
    key_list = []
    for k in weight_dict:
        key_list.append(k)
    with Path('para_key_mamba_3b.txt').open('w') as w_f:
        w_f.write('\n'.join(key_list))
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    return model, tokenizer

if __name__ == '__main__':
    # model, tokenizer = load_ckpts('3B', 'fully_qat_record/mamba2_3b_1_3B_1bit_amber/ckpt-11', 'torch', True, ['lm_head'], 'column')
    with Path('data/tokenized_alpacas.json').open('r', encoding='utf-8') as r_f:
        data = json.load(r_f)
    for k in data:
        print(k)
        print(len(data[k]))
    # print(model)
    # print(tokenizer)
    # inputs = tokenizer("Hey how are you doing?", return_tensors="pt")
    # inputs = {k:v.to('cuda') for k, v in inputs.items()}
    # print(model(**inputs).logits)

# tokenizer = AutoTokenizer.from_pretrained("pretrained/mamba-2.8b-hf")
# inputs = tokenizer("Hey how are you doing?", return_tensors="pt")
# inputs = {k:v.to('cuda') for k, v in inputs.items()}

# # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# # config = MambaConfig.from_pretrained('state-spaces/mamba-2.8b-hf')
# # config = MambaConfig()
# # print(config)
# # model = MambaLMHeadModel(config).to('cuda')
# # for k, v in model.named_parameters():
# #     print(k, v.shape)
# # for k, v in model.named_modules():
# #     print(k, type(v))
# # print(model(**inputs).logits)

# # model = MambaForCausalLM.from_pretrained("pretrained/mamba2-2.7b")
# # for k, v in model.named_parameters():
# #     print(k, v.shape)
# # print(model(**inputs).logits)


# # from transformers import LlamaForCausalLM
# # llama = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype='auto')
# # print(llama.model.layers[0].module)

# # record = []
# # model = MambaLMHeadModel.from_pretrained("pretrained/mamba2-2.7b").to('cuda')
# # for k, v in model.named_parameters():
# #     print(k, str(v))
# #     # record[k] = str(v)
# #     record.append(k)
# #     # record.append(str(v))
    
# # with Path('record_lmhead.txt').open('w') as w_f:
# #     w_f.write('\n'.join(record))


# record = {
#     "norm": 0,
#     "embedding": 0,
#     "dt_bias": 0,
#     "A_log": 0,
#     "D": 0,
#     "in_proj": 0,
#     "conv1d": 0,
#     "out_proj": 0,
# }
# model = MambaLMHeadModel.from_pretrained("pretrained/mamba2-2.7b").to('cuda')
# for k, v in model.named_parameters():
#     for kk in record.keys():
#         if kk in k:
#             record[kk] += v.numel()

# print(record)
# s = 0
# for k, v in record.items():
#     s += v
# record = {k: v/s for k, v in record.items()}
# print(record)

# # record = []
# # with Path(f'mamba2_3B.json').open('r') as r_f:
# #     _config = json.load(r_f)
# #     config = MambaConfig(**_config)
# # model = MambaForCausalLM(config=config)

# # for k, v in model.named_modules():
# #     record.append(k)
# #     record.append(str(v))
# #     print(k, v)
# # with Path('record_causal.txt').open('w') as w_f:
# #     w_f.write('\n'.join(record))


# # print(model.config)
# # print(model(**inputs).logits)

# # input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"]
# # out = model.generate(input_ids, max_new_tokens=100)
# # print(tokenizer.batch_decode(out))

# # record = {
# #     'param_size': [],
# #     'module_info': []
# # }






