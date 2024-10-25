import numpy as np
import torch
import json
from collections import Counter

from mamba_ssm.modules.block import Block
# from transformers.models.mamba.modeling_mamba import MambaBlock
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.mamba2 import Mamba2

from transformers import AutoTokenizer

from pathlib import Path

from qat.replace_module import (
    replace_with_learnable_binarylinear, 
    # replace_with_quantizelinear,
    check_para_state
)


def get_param_stat(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    
    param_dict = {}
    ckpt_plist = [p for p in ckpt_dir.iterdir() if p.suffix == '.bin']
    for p in ckpt_plist:
        _param_dict = torch.load(p)
        for k,v in _param_dict.items():
            param_dict[k] = v

    stat_res = {}
    for k in param_dict:
        if 'in_proj' in k or 'out_proj' in k:
            print(k)
            if 'weight' in k:
                scale = param_dict['.'.join(k.split('.')[:5]) + '.wscale']
                bias = param_dict['.'.join(k.split('.')[:5]) + '.wbias']
                _param = param_dict[k]
                param = scale * _param + bias
            else:
                param = param_dict[k]
            param_count = Counter(param.cpu().flatten().tolist())
            pos = torch.ge(param, 0).float().sum().item()
            neg = len(param.cpu().flatten().tolist()) - pos
            stat_res[k] = {
                'shape': param.shape,
                'pos': pos,
                'neg': neg,
                'min': param.min().item(),
                'max': param.max().item(),
                'mean': param.mean().item(),
                'std': param.std().item(),
                'unique': len(param_count),
                'count': dict(param_count)
            }
    
    save_p = Path(f"visual/{ckpt_dir.parent.name}_{ckpt_dir.name}.json")
    with save_p.open('w') as f:
        json.dump(stat_res, f, indent=4)

if __name__ == '__main__':
    dir_list = [p for p in Path("fully_qat_record/mamba2_3b_1_3B_1bit_amber").iterdir()]
    dir_list.sort(key=lambda x: int(x.name.split('-')[-1]), reverse=True)
    for ckpt_dir in dir_list:
        print(ckpt_dir.parent.name, ckpt_dir.name)
        get_param_stat(ckpt_dir)