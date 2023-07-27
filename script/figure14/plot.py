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
    methods = ["PyTorch", "PIT", "PyTorch-S"]
    sparse_ratio = [98, 96, 94, 92, 90, 85, 80, 70, 60, 50]
    sr_map = {'0.0209': 98,
        '0.0417': 96,
        '0.0625': 94,
        '0.0834': 92,
        '0.1042': 90,
        '0.1459': 85,
        '0.2084': 80,
        '0.2917': 70,
        '0.3959': 60,
        '0.5': 50
    }

    with open("results.txt") as f:
        data = [ii.strip().split(",") for ii in f.readlines()]
    g = {}
    for ii in data:
        mode, method, sr = ii[:3]
        sr = sr_map.get(sr, sr)
        method = method.replace("_32x1", "")
        if len(ii) == 5:
            latency, memory = [float(i) for i in ii[3:]]
            g[(mode, method, sr)] = [latency * 10, memory]
        else:
            latency, memory, convert = [float(i) for i in ii[3:]]
            if method != "PyTorch-S":
                g[(mode, method, sr)] = [latency * 10, memory]
            else:
                g[(mode, method, sr)] = [latency * 10, memory, convert * 10]
    nn_pruning_data_list, nn_pruning_data_32_list = [], []
    mode = "32x64"
    for sr in sparse_ratio:
        tmp = [sr]
        for method in methods:
            if method == "PyTorch-S":
                tmp += [0, 0, 0, 0, 0]
            if (mode, method, sr) not in g:
                if method == "PyTorch-S":
                    tmp += [0, 0, 0]
                else:
                    tmp += [0, 0]
                continue
            results = g[(mode, method, sr)]
            tmp += results
        nn_pruning_data_list.append(tmp)
    mode = "32x1"
    for sr in sparse_ratio:
        tmp = [sr]
        for method in methods:
            if method == "PyTorch-S":
                tmp += [0, 0, 0, 0, 0]
            if (mode, method, sr) not in g:
                if method == "PyTorch-S":
                    tmp += [0, 0, 0]
                else:
                    tmp += [0, 0]
                continue
            results = g[(mode, method, sr)]
            tmp += results
        nn_pruning_data_32_list.append(tmp)

    return nn_pruning_data_list, nn_pruning_data_32_list

def plot_nn_pruning_latency(nn_pruning_data_list, nn_pruning_data_32_list):
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

    keys = ["Pytorch", "Pytorch-S", "Pytorch-S Convert", "PIT"]

    ta = []
    for idx, data, label in zip(range(2), [nn_pruning_data_list, nn_pruning_data_32_list], ["32x64", "32x1"]):
        data = [ii for ii in data[::-1] if int(ii[0]) in [50, 80, 90, 94, 96, 98]]
        labels = [int(ii[0]) for jj, ii in enumerate(data)]

        pytorch_data = [ii[1] for jj, ii in enumerate(data)]
        triton_data = [ii[10] for jj, ii in enumerate(data)]
        triton_convert_data = [ii[12] for jj, ii in enumerate(data)]
        # print(triton_convert_data)
        ours_data = [ii[3] for jj, ii in enumerate(data)]
    #     print(ours_data[-1])
        x = []
        for i in range(len(labels)):
            x.append(i * 0.5+2.5)
        x = np.array(x)
        xx = x
        width = 0.12
    #     plt.xlim((x[0] - width * 4,x[-1]+width * 3))
        axs[idx].set_xlim((2.2,x[-1]+0.3))
    #     axs[idx].set_ylim((0,50))

        m1 = axs[idx].bar(x-1*width,pytorch_data, width, **styles['torch'])
        m2 = axs[idx].bar(x-0*width,triton_data, width, **styles['torch_s'])
        m3 = axs[idx].bar(x-0*width,triton_convert_data, width, **styles['torch_convert'])
    #     m4 = axs[idx].bar(x-0.5*width,turbo_data, width, **styles['turbo'], label='TurboTransformer')
    #     m5 = axs[idx].bar(x+0*width,deepspeed_data, width, **styles['ds'])
        m7 = axs[idx].bar(x+1*width,ours_data, width, **styles['pit'])
        # plt.bar(x+2*width,nmsparse_n32_50_data, width, edgecolor='black', color = 'white', hatch='oo', label='nmSPARSE-VW32')
        # plt.bar(x+3*width,nmsparse_n4k4_50_data, width, edgecolor='black', color = 'white', hatch='xx', label='nmSPARSE-BW4x4')
        ta.extend([m1, m2, m3, m7])

        if idx == 0:
            axs[idx].set_ylabel('Latency(ms)',)
    #     else:
    #         axs[idx].set_ylabel('GPU Memory(GB)',)
        axs[idx].set_xlabel("Sparsity(%)",)
        axs[idx].set_title(f"{label} Blocksizes",)
        axs[idx].set_xticks(x)
        axs[idx].set_xticklabels(labels)
    fig.legend(handles=ta, labels=keys, loc='upper center', bbox_to_anchor=(0.53, 1.15), ncol=4,frameon=False,)
    plt.savefig('figure14(a).pdf',bbox_inches='tight',pad_inches=0.0,dpi=1000)

    plt.show()

def plot_nn_pruning_memory(nn_pruning_data_list, nn_pruning_data_32_list):
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

    keys = ["Pytorch", "Pytorch-S", "PIT"]

    ta = []
    for idx, data, label in zip(range(2), [nn_pruning_data_list, nn_pruning_data_32_list], ["32x64", "32x1"]):
        data = [ii for ii in data[::-1] if int(ii[0]) in [50, 80, 90, 94, 96, 98]]
        labels = [int(ii[0]) for jj, ii in enumerate(data)]

        pytorch_data = [ii[2] for jj, ii in enumerate(data)]
        triton_data = [ii[11] for jj, ii in enumerate(data)]
        triton_convert_data = [ii[12] for jj, ii in enumerate(data)]
        ours_data = [ii[4] for jj, ii in enumerate(data)]

    #     print(ours_data[-1])
        x = []
        for i in range(len(labels)):
            x.append(i * 0.5+2.5)
        x = np.array(x)
        xx = x
        width = 0.12
    #     plt.xlim((x[0] - width * 4,x[-1]+width * 3))
        axs[idx].set_xlim((2.2,x[-1]+0.3))
    #     axs[idx].set_ylim((0,50))

        m1 = axs[idx].bar(x-1*width,pytorch_data, width, **styles['torch'])
        m2 = axs[idx].bar(x-0*width,triton_data, width, **styles['torch_s'])
    #     m3 = axs[idx].bar(x-0*width,triton_convert_data, width, **styles['torch_convert'])
    #     m4 = axs[idx].bar(x-0.5*width,turbo_data, width, **styles['turbo'], label='TurboTransformer')
    #     m5 = axs[idx].bar(x+0*width,deepspeed_data, width, **styles['ds'])
        m7 = axs[idx].bar(x+1*width,ours_data, width, **styles['pit'])
        # plt.bar(x+2*width,nmsparse_n32_50_data, width, edgecolor='black', color = 'white', hatch='oo', label='nmSPARSE-VW32')
        # plt.bar(x+3*width,nmsparse_n4k4_50_data, width, edgecolor='black', color = 'white', hatch='xx', label='nmSPARSE-BW4x4')
        ta.extend([m1, m2, m7])

        if idx == 0:
            axs[idx].set_ylabel('GPU Memory(GB)',)
    #     else:
    #         axs[idx].set_ylabel('GPU Memory(GB)',)
        axs[idx].set_xlabel("Sparsity(%)",)
        axs[idx].set_title(f"{label} Blocksizes",)
        axs[idx].set_xticks(x)
        axs[idx].set_xticklabels(labels)
    fig.legend(handles=ta, labels=keys, loc='upper center', bbox_to_anchor=(0.53, 1.15), ncol=4,frameon=False,)
    plt.savefig('figure14(b).pdf',bbox_inches='tight',pad_inches=0.0,dpi=1000)

    plt.show()


if __name__ == '__main__':
    nn_pruning_data_list, nn_pruning_data_32_list = load_results()
    plot_nn_pruning_latency(nn_pruning_data_list, nn_pruning_data_32_list)
    plot_nn_pruning_memory(nn_pruning_data_list, nn_pruning_data_32_list)
