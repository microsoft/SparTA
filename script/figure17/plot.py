import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SPARSITY_LIST = [0.5, 0.9, 0.95, 0.99]


def read_latency_from_file(path):
    with open(path) as f:
        log = f.readlines()
    for line in log:
        if 'latency' in line.lower() or 'time' in line.lower():
            return float(re.findall('\d+\.\d+', line)[0])


def read_pytorch_s_data(tile_size):
    return [
        read_latency_from_file(f'./log/{tile_size}_{sparsity}.log')
        for sparsity in SPARSITY_LIST
    ]


def read_pit_data(tile_size):
    df = pd.read_csv('./convert.csv')
    df = df[(df['block_h'] == tile_size) & (df['block_w'] == tile_size)][['sparsity', 't_avg']]
    return [
        df[df['sparsity'] == sparsity]['t_avg'].item()
        for sparsity in SPARSITY_LIST
    ]


plt.rc('font', size=30) #controls default text size
plt.rc('axes', titlesize=30) #fontsize of the title
plt.rc('axes', labelsize=30) #fontsize of the x and y labels
plt.rc('xtick', labelsize=30) #fontsize of the x tick labels
plt.rc('ytick', labelsize=30) #fontsize of the y tick labels
plt.rc('legend', fontsize=30) #fontsize of the legend

fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 4))

keys = ["PyTorch-S", "PIT"]

for idx, label in enumerate(["1x1", "16x16", "32x32"]):
    labels = [int(x) for x in SPARSITY_LIST]

    pytorch_s_data = read_pytorch_s_data(label.split('x')[0])
    ours_data = read_pit_data(int(label.split('x')[0]))

    x = []
    for i in range(len(labels)):
        x.append(i * 0.4+2.5)
    x = np.array(x)
    width = 0.12
    axs[idx].set_xlim((2.2,x[-1]+0.2))
#     axs[idx].set_ylim((0,50))

    axs[idx].bar(x-0.5*width,pytorch_s_data, width, edgecolor='black', color = 'white', hatch='\\', label='PyTorch-S')
    axs[idx].bar(x+0.5*width,ours_data, width, edgecolor='black', color = 'white', hatch='--', label='Spider')
    # plt.bar(x+2*width,nmsparse_n32_50_data, width, edgecolor='black', color = 'white', hatch='oo', label='nmSPARSE-VW32')
    # plt.bar(x+3*width,nmsparse_n4k4_50_data, width, edgecolor='black', color = 'white', hatch='xx', label='nmSPARSE-BW4x4')

    if idx == 0:
        axs[idx].set_ylabel('Latency(ms)')
    if idx == 1:
        axs[idx].set_xlabel("Sparsity(%)")
    axs[idx].set_title(f"Tile Size = {label}")
    axs[idx].set_xticks(x)
    axs[idx].set_xticklabels(labels)
fig.legend(labels=keys, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5,frameon=False)
plt.savefig('figure17.pdf',bbox_inches='tight',pad_inches=0.0,dpi=1000)
