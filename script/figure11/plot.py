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
    methods = ["PyTorch", "PIT", "PyTorch-S", "Longformer-S", "DeepSpeed"]
    models = ["longformer-base-2048", "longformer-large-2048", "longformer-base-4096", "longformer-large-4096"]
    model_map = {
        "longformer-base-4096/_2048": "longformer-base-2048",
        "longformer-large-4096/_2048": "longformer-large-2048",
        "longformer-base-4096/_4096": "longformer-base-4096",
        "longformer-large-4096/_4096": "longformer-large-4096",
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
            g[(model, method)] = [latency * 10, memory, convert * 10]
    longformer_data_list = []
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
        longformer_data_list.append(tmp)

    return longformer_data_list

def plot_longformer(longformer_data_list):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    plt.rc('font', size=40) #controls default text size
    plt.rc('axes', titlesize=40) #fontsize of the title
    plt.rc('axes', labelsize=40) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=40) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=40) #fontsize of the y tick labels
    plt.rc('legend', fontsize=40) #fontsize of the legend

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(20, 5))

    keys = ["Pytorch", "Pytorch-S", "Pytorch-S Convert", "Longformer-S", "DeepSpeed", "PIT"]

    ta = []
    for idx, label in zip(range(2), ["Latency", "GPU Memory"]):

        labels = [ii[0].replace("longformer-","") for ii in longformer_data_list]
        if idx == 0:
            pytorch_data = [ii[1] for ii in longformer_data_list]
            triton_data = [ii[5] for ii in longformer_data_list]
            triton_convert_data = [ii[7] for ii in longformer_data_list]
            ours_data = [ii[3] for ii in longformer_data_list]
            pytorch_sparse_data = [ii[8] for ii in longformer_data_list]
            ds_data = [ii[10] for ii in longformer_data_list]
        else:
            pytorch_data = [ii[2] for ii in longformer_data_list]
            triton_data = [ii[6] for ii in longformer_data_list]
            ours_data = [ii[4] for ii in longformer_data_list]
            triton_convert_data = [0 for ii in longformer_data_list]
            pytorch_sparse_data = [ii[9] for ii in longformer_data_list]
            ds_data = [ii[11] for ii in longformer_data_list]


    #     print(ours_data[-1])
        x = []
        for i in range(len(labels)):
            x.append(i * 0.6+2.5)
        x = np.array(x)
        xx = x
        width = 0.09
    #     plt.xlim((x[0] - width * 4,x[-1]+width * 3))
        axs[idx].set_xlim((x[0] - width * 3,x[-1]+width * 3))
    #     axs[idx].set_ylim((0,50))
        m1 = axs[idx].bar(x-2*width,pytorch_data, width, **styles['torch'])
        m2 = axs[idx].bar(x-1*width,triton_data, width, **styles['torch_s'])
        m3 = axs[idx].bar(x-1*width,triton_convert_data, width, **styles['torch_convert'])
        m4 = axs[idx].bar(x-0*width,pytorch_sparse_data, width, **styles['longformer_s'])
        m5 = axs[idx].bar(x+1*width,ds_data, width, **styles['ds'])
        m7 = axs[idx].bar(x+2*width,ours_data, width, **styles['pit'])

        # plt.bar(x+2*width,nmsparse_n32_50_data, width, edgecolor='black', color = 'white', hatch='oo', label='nmSPARSE-VW32')
        # plt.bar(x+3*width,nmsparse_n4k4_50_data, width, edgecolor='black', color = 'white', hatch='xx', label='nmSPARSE-BW4x4')
        ta.extend([m1, m2, m3, m4, m5, m7])
        for a, b in [(x-2*width,pytorch_data), (x-1*width,triton_data), (x+0*width,pytorch_sparse_data), (x+1*width,triton_data),(x+2*width,ours_data)]:
        #     print(a, b)
            for ii, jj in zip(a, b):
                if jj == 0:
                    axs[idx].scatter(ii, jj+(15 if idx == 0 else 0.8), s=100, color="black", marker="x")

        if idx == 0:
            axs[idx].set_ylabel('Latency(ms)',)
        else:
            axs[idx].set_ylabel('GPU Memory(GB)',)
        axs[idx].set_xlabel("Backbone & Seq Len.")
    #     axs[idx].set_title(f"{label}",)
        axs[idx].set_xticks(x)
        axs[idx].set_xticklabels(labels, fontsize=35)
    fig.legend(handles=ta, labels=keys, loc='upper center', bbox_to_anchor=(0.52, 1.4), ncol=4,frameon=False,)
    plt.savefig('figure11.pdf',bbox_inches='tight',pad_inches=0.0,dpi=1000)

    plt.show()


if __name__ == '__main__':
    longformer_data_list = load_results()
    plot_longformer(longformer_data_list)
