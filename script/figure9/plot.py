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
    'mega':{'edgecolor':cmap_colors[0], 'color' : 'white','hatch':'+', "label": "MegeBlocks"},
    'turbo':{'edgecolor':cmap_colors[0], 'color' : 'white','hatch':'*', "label": "TurboTransformer"},
    'pit': {'edgecolor':cmap_colors[0], 'color' : 'white', 'hatch':'--', "label": "PIT"},
}

def load_results():
    models = ["facebook/opt-13b", "facebook/opt-30b"]
    methods = ["PyTorch", "PyTorch-S", "PIT", "Turbo", "DeepSpeed", "PIT w/o activation"]
    with open("results.txt") as f:
        data = [ii.strip().split(",") for ii in f.readlines()]
    g = {}
    for ii in data:
        model, method = ii[:2]
        if len(ii) == 4:
            latency, memory = [float(i) for i in ii[2:]]
            g[(model, method)] = [latency * 5, memory]
        else:
            latency, memory, convert = [float(i) for i in ii[2:]]
            g[(model, method)] = [latency * 5, memory, convert * 5]
    opt_data_list = []
    for model in models:
        tmp = [model.split("/")[1].upper()]
        for method in methods:
            if method == "Turbo":
                tmp += [0, 0]
                continue
            if (model, method) not in g:
                if method == "PyTorch-S":
                    tmp += [0, 0, 0]
                else:
                    tmp += [0, 0]
            else:
                tmp += g.get((model, method), [0, 0])
            if method == "DeepSpeed":
                tmp.append("")
        opt_data_list.append(tmp)
    return opt_data_list

def plot_opt(opt_data_list):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    plt.rc('font', size=30) #controls default text size
    plt.rc('axes', titlesize=30) #fontsize of the title
    plt.rc('axes', labelsize=30) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=30) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=30) #fontsize of the y tick labels
    plt.rc('legend', fontsize=30) #fontsize of the legend

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15, 5))

    keys = ["Pytorch", "Pytorch-S", "Pytorch-S Convert", "DeepSpeed", "PIT w/o activation", "PIT"]

    ta = []
    for idx, label in zip(range(2), ["Latency", "GPU Memory"]):
        labels = [ii[0] for ii in opt_data_list]
        if idx == 0:
            pytorch_data = [ii[1] for ii in opt_data_list]
            triton_data = [ii[3] for ii in opt_data_list]
            triton_convert_data = [ii[5]for ii in opt_data_list]
            ours_data = [ii[6] for ii in opt_data_list]
            turbo_data = [0 for ii in opt_data_list]
            ours_wo_activation_data = [ii[13] for ii in opt_data_list]
            deepspeed_data = [ii[10] if ii[10] else 0 for ii in opt_data_list]
        else:
            pytorch_data = [ii[2] for ii in opt_data_list]
            triton_data = [ii[4] for ii in opt_data_list]
            triton_convert_data = [0 for ii in opt_data_list]
            ours_data = [ii[7] for ii in opt_data_list]
            turbo_data = [0 for ii in opt_data_list]
            ours_wo_activation_data = [ii[14] for ii in opt_data_list]
            deepspeed_data = [ii[11] if ii[11] else 0 for ii in opt_data_list]
    #     print(ours_data[-1])
        x = []
        for i in range(len(labels)):
            x.append(i * 0.3+2.5)
        x = np.array(x)
        xx = x
        width = 0.05
    #     plt.xlim((x[0] - width * 4,x[-1]+width * 3))
        axs[idx].set_xlim((2.35,x[-1]+0.15))
    #     axs[idx].set_ylim((0,50))
        for a, b in [(x-2.5*width,pytorch_data), (x-1.5*width,triton_data),(x+0.5*width,deepspeed_data), (x+1.5*width,ours_wo_activation_data), (x+2.5*width,ours_data)]:
        #     print(a, b)
            for ii, jj in zip(a, b):
                if jj == 0:
                    if idx == 0:
                        axs[idx].scatter(ii, jj+50, s=100, color="black", marker="x")
                    else:
                        axs[idx].scatter(ii, jj+0.5, s=100, color="black", marker="x")
        m1 = axs[idx].bar(x-1.5*width,pytorch_data, width, **styles['torch'])
        m2 = axs[idx].bar(x-0.5*width,triton_data, width, **styles['torch_s'])
        m3 = axs[idx].bar(x-0.5*width,triton_convert_data, width, **styles['torch_convert'])
    #     m4 = axs[idx].bar(x-0.5*width,turbo_data, width, **styles['turbo'], label='TurboTransformer')
        m5 = axs[idx].bar(x+0.5*width,deepspeed_data, width, **styles['ds'])
        m6 = axs[idx].bar(x+1.5*width,ours_wo_activation_data, width, **{"linestyle": "--", 'edgecolor':cmap_colors[0], 'color' : 'white', "label": "PIT"})
        m7 = axs[idx].bar(x+1.5*width,ours_data, width, **styles['pit'])
        # plt.bar(x+2*width,nmsparse_n32_50_data, width, edgecolor='black', color = 'white', hatch='oo', label='nmSPARSE-VW32')
        # plt.bar(x+3*width,nmsparse_n4k4_50_data, width, edgecolor='black', color = 'white', hatch='xx', label='nmSPARSE-BW4x4')
        ta.extend([m1, m2, m3, m5, m6, m7])

        if idx == 0:
            axs[idx].set_ylabel('Latency(ms)',)
        else:
            axs[idx].set_ylabel('GPU Memory(GB)',)
        axs[idx].set_xlabel("Models",)
    #     axs[idx].set_title(f"{label}",)
        axs[idx].set_xticks(x)
        axs[idx].set_xticklabels(labels)
    fig.legend(handles=ta, labels=keys, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3,frameon=False,)
    plt.savefig('figure9.pdf', bbox_inches='tight', pad_inches=0.0, dpi=1000)

    plt.show()

if __name__ == '__main__':
    opt_data_list = load_results()
    plot_opt(opt_data_list)
