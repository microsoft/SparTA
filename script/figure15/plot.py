import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from brokenaxes import brokenaxes


SPARSITY_LIST = [0.5, 0.9, 0.95, 0.99]


def read_latency_from_file(path):
    with open(path) as f:
        log = f.readlines()
    for line in log:
        if 'latency' in line.lower() or 'time' in line.lower():
            return float(re.findall('\d+\.\d+', line)[0])


def read_data_by_block_size(prefix, block_size):
    return [
        read_latency_from_file(f'./log/{prefix}_{sparsity}_{block_size}.log')
        for sparsity in SPARSITY_LIST
    ]


def read_sparta_data_by_block_size(block_size):
    df = pd.read_csv('./sparta_results.csv')
    return df[df['block_size'] == block_size]['latency'].to_list()


plt.rc('font', size=30) #controls default text size
plt.rc('axes', titlesize=30) #fontsize of the title
plt.rc('axes', labelsize=30) #fontsize of the x and y labels
plt.rc('xtick', labelsize=30) #fontsize of the x tick labels
plt.rc('ytick', labelsize=30) #fontsize of the y tick labels
plt.rc('legend', fontsize=30) #fontsize of the legend

fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(18, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})

keys = ["cuSPARSE", "Spunik", "OpenAI Block", "SparTA", "PIT"]

for idx, label in enumerate(["32x1", "1x64", "32x64"]):
    bh, bw = [int(x) for x in label.split('x')]
    cuSPARSE_data = read_data_by_block_size('cusparse', f'{bh}_{bw}')
    spunik_data = read_data_by_block_size('sputnik', f'{bh}_{bw}')
    openai_data = read_data_by_block_size('openai', f'{bh}_{bw}')
    ours_data = read_data_by_block_size('pit', f'{bh}_{bw}')
    sparta_data = read_sparta_data_by_block_size(label)
    y = max(cuSPARSE_data)

    ax1 = axs[0][idx]
    ax2 = axs[1][idx]
    
    x = []
    for i in range(len(SPARSITY_LIST)):
        x.append(i * 0.75+2.5)
    x = np.array(x)
    width = 0.12
#     axs[idx] = brokenaxes(ylims=((0, 50), (y-10, y+5)), subplot_spec=axs[idx])
#     axs[idx].set_xlim((2.2,x[-1]+0.4))
#     axs[idx].set_ylim((0,50))
    if idx == 0:
        ax1.set_ylim(15, y + 5)  # outliers only
        ax2.set_ylim(0, 15)  # most of the data
    elif idx == 1:
        ax1.set_ylim(20, y + 5)  # outliers only
        ax2.set_ylim(0, 20)  # most of the data
    else:
        ax1.set_ylim(20, y + 5)  # outliers only
        ax2.set_ylim(0, 20)  # most of the data
    ax1.set_xlim((2.2,x[-1]+0.4))
    ax2.set_xlim((2.2,x[-1]+0.4))

    ax1.bar(x-2*width,cuSPARSE_data, width, edgecolor='black', color = 'white', label='cuSPARSE')
    ax1.bar(x-1*width,spunik_data, width, edgecolor='black', color = 'white', hatch='\\\\\\', label='Spunik')
    ax1.bar(x-0*width,openai_data, width, edgecolor='black', color = 'white', hatch='oo', label='OpenAI Block')
    ax1.bar(x+1*width,sparta_data, width, edgecolor='black', color = 'white', hatch='xx', label='SparTA')
    ax1.bar(x+2*width,ours_data, width, edgecolor='black', color = 'white', hatch='--', label='Spider')
    
    a1 = ax2.bar(x-2*width,cuSPARSE_data, width, edgecolor='black', color = 'white', label='cuSPARSE')
    a2 = ax2.bar(x-1*width,spunik_data, width, edgecolor='black', color = 'white', hatch='\\\\\\', label='Spunik')
    a3 = ax2.bar(x-0*width,openai_data, width, edgecolor='black', color = 'white', hatch='oo', label='OpenAI Block')
    a4 = ax2.bar(x+1*width,sparta_data, width, edgecolor='black', color = 'white', hatch='xx', label='SparTA')
    a5 = ax2.bar(x+2*width,ours_data, width, edgecolor='black', color = 'white', hatch='--', label='Spider')
    # plt.bar(x+2*width,nmsparse_n32_50_data, width, edgecolor='black', color = 'white', hatch='oo', label='nmSPARSE-VW32')
    # plt.bar(x+3*width,nmsparse_n4k4_50_data, width, edgecolor='black', color = 'white', hatch='xx', label='nmSPARSE-BW4x4')

    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.xaxis.tick_top()
#     ax1.set([])
    ax1.tick_params(top=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=18,
                  linestyle="none", color='k', mec='k', mew=2, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    if idx == 0:
        ax2.set_ylabel('Latency(ms)')
    if idx == 1:
        ax2.set_xlabel("Sparsity(%)")
    ax1.set_title(f"Tile Size = {label}")
    ax1.set_xticks([])
    ax2.set_xticks(x)
    ax2.set_xticklabels(SPARSITY_LIST)
fig.legend(handles=[a1, a2, a3, a4, a5], labels=keys, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5,frameon=False)
plt.savefig('figure15.pdf',bbox_inches='tight',pad_inches=0.0,dpi=1000)
