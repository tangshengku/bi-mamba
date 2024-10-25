import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np
# from utils import load_json

import json

ppl = {
   "llama7": {
        'wikitext2':5.6770,
        'ptb':41.1509,
        'c4':7.3435
   }
}

baseline_color = {
    "rand": 'r',
    "llama7": 'g',
    "PB_LLM_10": 'b',
    "PB_LLM_30": 'c'
}
tasks_res = {
   "rand": {
        'boolq': 0.5,
        'piqa': 0.5,
        "hellaswag": 0.25,
        "winogrande": 0.5,
        "arc_easy": 0.25,
        "arc_challenge": 0.25,
        "openbookqa": 0.25
    },
    "llama7": {
        'boolq': 0.768,
        'piqa': 0.793,
        "hellaswag": 0.761,
        "winogrande": 0.700,
        "arc_easy": 0.730,
        "arc_challenge": 0.480,
        "openbookqa": 0.576,
        "mean_acc": 0.687
    },
    "PB_LLM_10": {
        'boolq': 0.689,
        'piqa': 0.678,
        "hellaswag": 0.681,
        "winogrande": 0.674,
        "arc_easy": 0.587,
        "arc_challenge": 0.429,
        "openbookqa": 0.506,
        "mean_acc": 0.606
    },
    "PB_LLM_30": {
        'boolq': 0.757,
        'piqa': 0.780,
        "hellaswag": 0.743,
        "winogrande": 0.697,
        "arc_easy": 0.690,
        "arc_challenge": 0.456,
        "openbookqa": 0.558,
        "mean_acc": 0.669
    }
}


def merge(tag):
    base_dir = Path('qat_record/eval_res')
    candi_list = [p for p in base_dir.iterdir() if tag == '_'.join(p.stem.split('_')[:-1])]
    res = {}
    for p in candi_list:
        r = load_json(p)
        res.update(r)
    order_ks = sorted([k for k in res], key=lambda x: int(x.split('_')[-1]))
    result = {k: res[k] for k in order_ks}
    print(result)
    return result



def plot1():
    x = list(range(32))
    y = []

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_0-1-2-3.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [0,1,2,3]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_4-5-6-7.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [4,5,6,7]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_8-9-10-11.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [8,9,10,11]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_12-13-14.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [12,13,14]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_15-16.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [15,16]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_17-18.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [17,18]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_19-20.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [19,20]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_21-22.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [21,22]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_23-24.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [23,24]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_25.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [25]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_26.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [26]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_27.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [27]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_28.json')
    # with p.open('r', encoding='utf-8') as f:
    #     res = json.load(f)
    # for i in [28]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    p = Path('eval_result/bf_mamba2_3b_1_3B_1bit_amber_29-30-31.json')
    with p.open('r', encoding='utf-8') as f:
        res = json.load(f)

    for k, v in res[f"ckpt-31"]["results"].items():
        print(k, v)

    print(res[f"ckpt-31"]["wikitext2"])
    print(res[f"ckpt-31"]["ptb"])
    print(res[f"ckpt-31"]["c4"])
    # print(res[f"ckpt-31"]["results"])
    # for i in [29,30,31]:
    #     y.append(res[f"ckpt-{i}"]["results"]["hellaswag"]["acc_norm,none"])

    # plt.plot(x, y, alpha=0.7, label = '3B')
    # plt.legend(fontsize=18)
    # plt.savefig(f'figures/task.jpg', dpi=300)

    

def plot3():
    x = list(range(15))
    p = Path('eval_result/bf_mamba2_1.3b_1_1.3B_1bit_amber_0-1-2.json')
    with p.open('r', encoding='utf-8') as f:
        res = json.load(f)
    y = []
    for i in x:
        y.append(res[f"ckpt-{i}"]["wikitext2"])

    plt.plot(x, y, alpha=0.7, label = '1.3B')
    x = [0, 1, 2, 3, 4]
    p = Path('eval_result/bf_mamba2_780m_1_780M_1bit_amber_0-1-2-3-4.json')
    with p.open('r', encoding='utf-8') as f:
        res = json.load(f)
    y = []
    for i in x:
        y.append(res[f"ckpt-{i}"]["wikitext2"])

    plt.plot(x, y, alpha=0.7, label = '780M')
    plt.legend(fontsize=18)
    plt.savefig(f'figures/ppl.jpg', dpi=300)


def plot4():
    x = [10, 11, 12,13,14]
    y = [
        0.5464050985859391, 
        0.552180840470025,
        0.5530770762796255,
        0.5593507269468233,
        0.560246962756423
        ]
    plt.plot(x, y, alpha=0.7, label = '2.7B')
    plt.legend(fontsize=18)
    plt.savefig(f'figures/hellaswag.jpg', dpi=300)

    
    

    

# def plot_ppl_fig():
#     # tags = ['fb_2000_row_128', 'ft_2000_128_4', 'ag-ft_2000_128_4_ag2000', 'fb_5000_column_right']
#     # tags = ['fb_2000_row_128', 'ag-ft_2000_128_4_ag2000', 'fb_5000_column_right']
#     # tags = ['fb_2000_row_128', 'ag-2000_128_head', 'ag-2000_row_128_head_norm', 'ag-2000_row_128_head_norm_embed']
#     # tags = ['ag-2000_row_128_head_norm_embed', 'fb_5000_column_right', 'ag_column', 'ag_column_head_norm_embed']
#     # tags = ['ag_column_head_norm_embed', 'ag-2000_row_128_head_norm_embed', 'ag_single', 'ag_single_head_norm_embed', 'ag-column-all-ly0_att_F', 'ag-column-all-ly0_F']
#     # tags = ['workdir_130M']
#     tags = ['workdir_130M']
#     data = ['wikitext2','ptb']
#     # x = [i*100 for i in range(1, 21)]
#     # x = [i*100 for i in range(2, 21, 2)]
#     x = [i for i in range(0, 71, 2)]

#     all_res = {}
#     for tag in tags:
#         res = merge(tag)
#         all_res[tag] = res
    
#     for d in data:
#         plt.figure(figsize=(20, 12))
#         plt.rcParams['xtick.direction'] = 'in'
#         plt.rcParams['ytick.direction'] = 'in'
#         for tag in tags:
#             y = []
#             for i in x:
#                 if tag == 'fb_5000_column_right':
#                     ck_name = f"checkpoint-{i*2}"
#                 else:
#                     ck_name = f"ckpt_{i}"
#                 if ck_name in all_res[tag]:
#                     v = all_res[tag][ck_name][d]
#                     y.append(v)
#             print(y)
#             plt.plot(x[3:], y[3:], alpha=0.7, label=tag)
#         for baseline in ppl:
#             plt.plot(x[3:], ([ppl[baseline][d]]*len(x))[3:], alpha=0.7, color=baseline_color[baseline], label=baseline)
#         plt.title(d, fontsize=21)
#         plt.xlabel('train steps',fontsize=20)
#         plt.ylabel('ppl',fontsize=20)
#         plt.grid(linestyle='--')
#         plt.legend(fontsize=18)
#         plt.savefig(f'figures_1bit/{d}.jpg', dpi=300)


if __name__ == '__main__':
  plot1()

#   plot3()
#   plot4()
