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
    methods = ["PyTorch", "PyTorch-S", "PIT",  "Turbo", "DeepSpeed"]
    models = ["OPT-125M", "OPT-350M", "OPT-1.3B"]
    model_map = {
        "facebook/opt-125m": "OPT-125M",
        "facebook/opt-350m": "OPT-350M",
        "facebook/opt-1.3b": "OPT-1.3B",
    }

    with open("results.txt") as f:
        data = [ii.strip().split(",") for ii in f.readlines()]
    g = {}
    for ii in data:
        model, method = ii[:2]
        model = model_map.get(model, model)
        if len(ii) == 4:
            latency, memory = [float(i) for i in ii[2:]]
            g[(model, method)] = [latency * 10, memory]
        else:
            latency, memory, convert = [float(i) for i in ii[2:]]
            if method != "PyTorch-S":
                g[(model, method)] = [latency * 10, memory]
            else:
                g[(model, method)] = [latency * 10, memory, convert * 10]
    opt_data_list = []
    for model in models:
        tmp = [model]
        for method in methods:
            if (model, method) not in g:
                if method == "PyTorch-S":
                    tmp += [0, 0, 0]
                else:
                    tmp += [0, 0]
                continue
            results = g[(model, method)]
            tmp += results
        opt_data_list.append(tmp)

    return opt_data_list

def plot_opt_training(opt_data_list):
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

    keys = ["Pytorch", "Pytorch-S", "Pytorch-S Convert", "DeepSpeed", "PIT"]

    ta = []
    for idx, label in zip(range(2), ["Latency", "GPU Memory"]):
        labels = [ii[0] for ii in opt_data_list]
        if idx == 0:
            pytorch_data = [ii[1] for ii in opt_data_list]
            triton_data = [ii[3] for ii in opt_data_list]
            triton_convert_data = [ii[5]for ii in opt_data_list]
            ours_data = [ii[6] for ii in opt_data_list]
            deepspeed_data = [ii[10] if ii[10] else 0 for ii in opt_data_list]
        else:
            pytorch_data = [ii[2] for ii in opt_data_list]
            triton_data = [ii[4] for ii in opt_data_list]
            triton_convert_data = [0 for ii in opt_data_list]
            ours_data = [ii[7] for ii in opt_data_list]
            deepspeed_data = [ii[11] if ii[11] else 0 for ii in opt_data_list]

    #     print(ours_data[-1])
        x = []
        for i in range(len(labels)):
            x.append(i * 0.4+2.5)
        x = np.array(x)
        xx = x
        width = 0.07
    #     plt.xlim((x[0] - width * 4,x[-1]+width * 3))
        axs[idx].set_xlim((x[0] - width * 3,x[-1]+width * 3))
    #     axs[idx].set_ylim((0,50))
        m1 = axs[idx].bar(x-1.5*width,pytorch_data, width, **styles['torch'])
        m2 = axs[idx].bar(x-0.5*width,triton_data, width, **styles['torch_s'])
        m3 = axs[idx].bar(x-0.5*width,triton_convert_data, width, **styles['torch_convert'])
    #     m4 = axs[idx].bar(x-0.5*width,turbo_data, width, **styles['turbo'], label='TurboTransformer')
        m5 = axs[idx].bar(x+0.5*width,deepspeed_data, width, **styles['ds'])
        m7 = axs[idx].bar(x+1.5*width,ours_data, width, **styles['pit'])
        # plt.bar(x+2*width,nmsparse_n32_50_data, width, edgecolor='black', color = 'white', hatch='oo', label='nmSPARSE-VW32')
        # plt.bar(x+3*width,nmsparse_n4k4_50_data, width, edgecolor='black', color = 'white', hatch='xx', label='nmSPARSE-BW4x4')
        ta.extend([m1, m2, m3, m5, m7])

        if idx == 0:
            axs[idx].set_ylabel('Latency(ms)',)
        else:
            axs[idx].set_ylabel('GPU Memory(GB)',)
        axs[idx].set_xlabel("Models",)
    #     axs[idx].set_title(f"{label}",)
        axs[idx].set_xticks(x)
        axs[idx].set_xticklabels(labels)
    fig.legend(handles=ta, labels=keys, loc='upper center', bbox_to_anchor=(0.52, 1.3), ncol=4,frameon=False,)
    plt.savefig('figure13.pdf',bbox_inches='tight',pad_inches=0.0,dpi=1000)

    plt.show()

if __name__ == '__main__':
    opt_data_list = load_results()
    plot_opt_training(opt_data_list)
