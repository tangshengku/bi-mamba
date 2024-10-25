import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np
# from utils import load_json

import json


def plot_wiki():
    x = [i*0.1 for i in range(1, 10)]
    y = [
            11.214934349060059,
            17.393993377685547,
            2187.596923828125,
            2752.11767578125,
            16686.58984375,
            28246.3671875,
            42722.515625,
            91142.7421875,
            187445.0625
        ]
    x2 = [0.951]
    y2 = [9.1]

    plt.figure(figsize=(16, 10))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.plot(x, y, label='GPTQ (2-bits)', marker='o')
    plt.plot(x2, y2, label='Bi-Mamba2 (1-bits)', marker='*')

    plt.yscale('log')

    plt.xlabel('Quantizztion Ratio',fontsize=20)
    plt.ylabel('log(PPL)',fontsize=20)

    plt.grid(linestyle='--')
    plt.legend(fontsize=18)

    plt.savefig(f'test.jpg', dpi=300)
    # plt.savefig(f'test.jpg', dpi=300)

if __name__ == '__main__':
    plot_wiki()