import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
sns.set_style('white')
sns.set_theme("poster", style="ticks", font_scale=1.2)
plt.rc('font', family="Times New Roman")

import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
sns.set_style('white')
sns.set_theme("poster", style="ticks", font_scale=1.2)
plt.rc('font', family="Times New Roman")
# cmap_colors = ['#fbb4ae', '#b3cde3', '#decbe4', '#fed9a6', '#e5d8bd', '#fddaec', '#f2f2f2']
cmap_colors = [
"black",
'firebrick',
    '#1b9e77',
 '#d95f02',
 '#7570b3',
#  '#e7298a',
 '#66a61e',
 '#e6ab02',
 '#a6761d',
 '#666666',
 '#666666']
styles={
    'torch':{'edgecolor':cmap_colors[0], 'color' : 'white', 'hatch':'//', "label": "PyTorch"},
    'torch_s':{'edgecolor':cmap_colors[0], 'color' : 'white', 'hatch':'\\', "label": "PyTorch-S"},
    'torch_convert':{'edgecolor':cmap_colors[0], 'color' : 'white', 'hatch':'xx', "label": "PyTorch-S Convert"},
    'ds':{'edgecolor':cmap_colors[0], 'color' : 'white','hatch':'oo', "label": "DeepSpeed"},
    'tutel':{'edgecolor':cmap_colors[0], 'color' : 'white', 'hatch':'O', "label": "Tutel"},
    'longformer_s':{'edgecolor':cmap_colors[0], 'color' : 'white', 'hatch':'/o', "label": "Longformer-S"},
    'mega':{'edgecolor':cmap_colors[0], 'color' : 'white','hatch':'+', "label": "MegaBlocks"},
    'turbo':{'edgecolor':cmap_colors[0], 'color' : 'white','hatch':'*', "label": "TurboTransformer"},
    'pit': {'edgecolor':cmap_colors[0], 'color' : 'white', 'hatch':'--', "label": "PIT"},
}

def load_results():
    methods = ["PyTorch", "Tutel", "PIT", "DeepSpeed", "MegaBlocks", "PyTorch-S", "PIT w/o MoE"]
    keys = [("True", "32"), ("True", "8"), ("False", "32"), ("False", "8")]
    exp_nums = ["64", "128", "256"]
    with open("results.txt") as f:
        data = [ii.strip().split(",") for ii in f.readlines()]
    g = {}
    for ii in data:
        fp16, exp_num, bsz, method = ii[:4]
        if len(ii) == 6:
            latency, memory = [float(i) for i in ii[4:]]
            g[(fp16, bsz, exp_num, method)] = [latency * 10, memory]
        else:
            latency, memory, convert = [float(i) for i in ii[4:]]
            g[(fp16, bsz, exp_num, method)] = [latency * 10, memory, convert * 10]

    res = [[], [], [], []]
    for ii, (fp16, bsz) in enumerate(keys):
        for exp_num in exp_nums:
            tmp = [exp_num]
            for method in methods:
                if method == "PyTorch-S":
                    tmp.append(0)
                if (fp16, bsz, exp_num, method) not in g:
                    if method == "PyTorch-S":
                        tmp += [0, 0, 0]
                    else:
                        tmp += [0, 0]
                    continue
                tmp += g[(fp16, bsz, exp_num, method)]
            res[ii].append(tmp)
    return res

def plot_moe_latency(moe_bsz_32_data_list, moe_bsz_8_list, moe_bsz_fp32_32_data_list, moe_bsz_fp32_8_list):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    plt.rc('font', size=30) #controls default text size
    plt.rc('axes', titlesize=30) #fontsize of the title
    plt.rc('axes', labelsize=30) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=30) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=30) #fontsize of the y tick labels
    plt.rc('legend', fontsize=30) #fontsize of the legend

    fig, axs = plt.subplots(2, 4, constrained_layout=True, figsize=(30, 6))

    keys = ["Pytorch", "Pytorch-S", "Pytorch-S Convert", "Tutel", "DeepSpeed", "MegaBlocks", "PIT w/o Sparse MoE", "PIT"]

    ta = []
    for idx, data, label in zip(range(4), [moe_bsz_32_data_list, moe_bsz_8_list, moe_bsz_fp32_32_data_list, moe_bsz_fp32_8_list], ["32", "8", "32", "8"]):
        # data = [jj for ii, jj in enumerate(data) if ii > 2]    

        labels = [int(ii[0]) for jj, ii in enumerate(data)]

        forloop_data = [ii[1] for jj, ii in enumerate(data)]
        batchmatmul_data = [ii[3] for jj, ii in enumerate(data)]
        ours_data = [ii[5] for jj, ii in enumerate(data)]
        ds_data = [ii[7] for jj, ii in enumerate(data)]
        megablocks_data = [ii[9] for jj, ii in enumerate(data)]
        triton_data = [ii[12] for jj, ii in enumerate(data)]
        triton_convert = [ii[14] for jj, ii in enumerate(data)]
        ours_wo_data = [ii[15] for jj, ii in enumerate(data)]
        
        
        ax1 = axs[0][idx]
        ax2 = axs[1][idx]
        
        x = []
        for i in range(len(labels)):
            x.append(i * 0.6+2.5)
        x = np.array(x)
        width = 0.08

        if idx == 0:
            ax1.set_ylim(100, 1000)  # outliers only
            ax2.set_ylim(0, 100)  # most of the data
        elif idx == 1:
            ax1.set_ylim(70, 600)  # outliers only
            ax2.set_ylim(0, 70)  # most of the data
        elif idx == 2:
            ax1.set_ylim(300, 4000)  # outliers only
            ax2.set_ylim(0,300)  # most of the data
        elif idx == 3:
            ax1.set_ylim(200, 2500)  # outliers only
            ax2.set_ylim(0, 200)  # most of the data
        ax1.set_xlim((2.2,x[-1]+0.3))
        ax2.set_xlim((2.2,x[-1]+0.3))

        ax1.bar(x-2.5*width,forloop_data, width, **styles['torch'])
        ax1.bar(x-1.5*width,triton_data, width, **styles['torch_s'])
        ax1.bar(x-1.5*width,triton_convert, width, **styles['torch_convert'])
        ax1.bar(x-0.5*width,batchmatmul_data, width, **styles['tutel'])
        ax1.bar(x+0.5*width,ds_data, width, **styles['ds'])
        ax1.bar(x+1.5*width,megablocks_data, width, **styles['mega'])
        
        ax1.bar(x+2.5*width,ours_wo_data, width, **{"linestyle": "--", 'edgecolor':cmap_colors[0], 'color' : 'white', "label": "PIT"})
        ax1.bar(x+2.5*width,ours_data, width, **styles['pit'])
        
        m1 = ax2.bar(x-2.5*width,forloop_data, width, **styles['torch'])
        m6 = ax2.bar(x-1.5*width,triton_data, width, **styles['torch_s'])
        m7 = ax2.bar(x-1.5*width,triton_convert, width, **styles['torch_convert'])
        m2 = ax2.bar(x-0.5*width,batchmatmul_data, width, **styles['tutel'])
        m3 = ax2.bar(x+0.5*width,ds_data, width, **styles['ds'])
        m4 = ax2.bar(x+1.5*width,megablocks_data, width, **styles['mega'])
        
        m8 = ax2.bar(x+2.5*width,ours_wo_data, width, **{"linestyle": "--", 'edgecolor':cmap_colors[0], 'color' : 'white', "label": "PIT"})
        m5 = ax2.bar(x+2.5*width,ours_data, width, **styles['pit'])

        ax1.spines["bottom"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax1.xaxis.tick_top()

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=18,
                    linestyle="none", color='k', mec='k', mew=2, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
        
        
        print(ours_data[-1])
        x = []
        for i in range(len(labels)):
            x.append(i * 0.6+2.5)
        x = np.array(x)
        width = 0.08
        ax2.set_xlim((2.2,x[-1]+0.3))

        for a, b in [(x-0.5*width, batchmatmul_data), (x+0.5*width,ds_data), (x+1.5*width,megablocks_data)]:
            for ii, jj in zip(a, b):
                if jj == 0:
                    if idx == 0:
                        ax2.scatter(ii, jj+60, s=100, color="black", marker="x")
                    elif idx == 3:
                        ax2.scatter(ii, jj+10, s=100, color="black", marker="x")
                    else:
                        ax2.scatter(ii, jj+15, s=100, color="black", marker="x")

        ta.extend([m1, m6, m7, m2, m3, m4, m8, m5])

        if idx == 0:
            ax2.set_ylabel('Latency(ms)',)
        ax2.set_xlabel("Expert Number",)
        ax1.set_title(f"Batch Size = {label}",)
        ax1.set_xticks([])
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax1.tick_params(top=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        ax1.xaxis.tick_top()
        ax1.tick_params(top=False)  # don't put tick labels at the top

        if idx == 0:
            ax2.text(.84, -1, 'FP16',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=ax2.transAxes)
        if idx == 1:
            ax2.text(-0.13, -1, 'Latency',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=ax2.transAxes)
        if idx == 2:
            ax2.text(.84, -1, 'FP32',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=ax2.transAxes)
        if idx == 3:
            ax2.text(-0.18, -1, 'Latency',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=ax2.transAxes)

    fig.legend(handles=ta, labels=keys, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=8,frameon=False,)
    plt.savefig('figure8(a).pdf', bbox_inches='tight', pad_inches=0.0, dpi=1000)

    plt.show()


def plot_moe_memory(moe_bsz_32_data_list, moe_bsz_8_list, moe_bsz_fp32_32_data_list, moe_bsz_fp32_8_list):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    plt.rc('font', size=30) #controls default text size
    plt.rc('axes', titlesize=30) #fontsize of the title
    plt.rc('axes', labelsize=30) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=30) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=30) #fontsize of the y tick labels
    plt.rc('legend', fontsize=30) #fontsize of the legend

    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(30, 6))

    keys = ["Pytorch", "Pytorch-S",  "Tutel", "DeepSpeed", "MegaBlocks", "PIT w/o Sparse MoE", "PIT"]

    ta = []
    for idx, data, label in zip(range(4), [moe_bsz_32_data_list, moe_bsz_8_list, moe_bsz_fp32_32_data_list, moe_bsz_fp32_8_list], ["32", "8", "32", "8"]):
        # data = [jj for ii, jj in enumerate(data) if ii > 2]    

        labels = [int(ii[0]) for jj, ii in enumerate(data)]

        forloop_data = [ii[2] for jj, ii in enumerate(data)]
        batchmatmul_data = [ii[4] for jj, ii in enumerate(data)]
        ours_data = [ii[6] for jj, ii in enumerate(data)]
        ds_data = [ii[8] for jj, ii in enumerate(data)]
        megablocks_data = [ii[10] for jj, ii in enumerate(data)]
        triton_data = [ii[13] for jj, ii in enumerate(data)]
        ours_wo_data = [ii[16] for jj, ii in enumerate(data)]
    #     triton_convert = [ii[15] for jj, ii in enumerate(data)]
        print(ours_data[-1])
        x = []
        for i in range(len(labels)):
            x.append(i * 0.6+2.5)
        x = np.array(x)
        width = 0.08
        axs[idx].set_xlim((2.2,x[-1]+0.4))
    #     axs[idx].set_ylim((0,50))
        for a, b in [(x-0.5*width, batchmatmul_data), (x+0.5*width,ds_data), (x+1.5*width,megablocks_data)]:
        #     print(a, b)
            for ii, jj in zip(a, b):
                if jj == 0:
                    if idx == 0:
                        axs[idx].scatter(ii, jj+2, s=100, color="black", marker="x")
                    elif idx == 3:
                        axs[idx].scatter(ii, jj+1.5, s=100, color="black", marker="x")
                    else:
                        axs[idx].scatter(ii, jj+1.2, s=100, color="black", marker="x")
        m1 = axs[idx].bar(x-2.5*width,forloop_data, width, **styles['torch'])
        m6 = axs[idx].bar(x-1.5*width,triton_data, width, **styles['torch_s'])
        m2 = axs[idx].bar(x-0.5*width,batchmatmul_data, width, **styles['tutel'])
        m3 = axs[idx].bar(x+0.5*width,ds_data, width, **styles['ds'])
        m4 = axs[idx].bar(x+1.5*width,megablocks_data, width, **styles['mega'])
        m8 = axs[idx].bar(x+2.5*width,ours_wo_data, width, **{"linestyle": "--", 'edgecolor':cmap_colors[0], 'color' : 'white', "label": "PIT"})
        m5 = axs[idx].bar(x+2.5*width,ours_data, width, **styles['pit'])
        ta.extend([m1, m6, m2, m3, m4, m8, m5])

        if idx == 0:
            axs[idx].set_ylabel('GPU Memory(GB)',)
        axs[idx].set_xlabel("Expert Number",)
        axs[idx].set_title(f"Batch Size = {label}",)
        axs[idx].set_xticks(x)
        axs[idx].set_xticklabels(labels)
        if idx == 0:
            axs[idx].text(.83, -0.4, 'FP16',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=axs[idx].transAxes)
        if idx == 1:
            axs[idx].text(-0.13, -0.4, 'GPU Memory',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=axs[idx].transAxes)
        if idx == 2:
            axs[idx].text(.83, -0.4, 'FP32',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=axs[idx].transAxes)
        if idx == 3:
            axs[idx].text(-0.13, -0.4, 'GPU Memory',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=axs[idx].transAxes)
    fig.legend(handles=ta, labels=keys, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=7,frameon=False,)
    plt.savefig('figure8(b).pdf', bbox_inches='tight', pad_inches=0.0, dpi=1000)

    plt.show()



if __name__ == '__main__':
    moe_bsz_32_data_list, moe_bsz_8_list, moe_bsz_fp32_32_data_list, moe_bsz_fp32_8_list = load_results()
    plot_moe_latency(moe_bsz_32_data_list, moe_bsz_8_list, moe_bsz_fp32_32_data_list, moe_bsz_fp32_8_list)
    plot_moe_memory(moe_bsz_32_data_list, moe_bsz_8_list, moe_bsz_fp32_32_data_list, moe_bsz_fp32_8_list)
