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
    methods = ["PyTorch", "PIT", "PyTorch-S", "CuSparse", "DeepSpeed", "Turbo"]
    datasets1 = ["mnli", "mrpc", "cola", "rte", "qqp", "sst2", "wnli", "qnli", "stsb"]
    datasets2 = ["imdb", "xsci.", "news"]
    d2_map = {
        "imdb_batch": "imdb",
        "multi_x_scrience_batch": "xsci.",
        "multi_news_batch": "news",
    }

    with open("results.txt") as f:
        data = [ii.strip().split(",") for ii in f.readlines()]
    g = {}
    for ii in data:
        dataset, method = ii[:2]
        dataset = dataset.replace(".pkl", "")
        dataset = d2_map.get(dataset, dataset)
        if len(ii) == 4:
            latency, memory = [float(i) for i in ii[2:]]
            g[(dataset, method)] = [latency * 10, memory]
        else:
            latency, memory, convert = [float(i) for i in ii[2:]]
            g[(dataset, method)] = [latency * 10, memory, convert * 10]
    seq_len_glue_data_list, seq_len_long_document_data_list = [], []
    
    for d in datasets1:
        tmp = [d]
        for method in methods:
            if (d, method) not in g:
                if method == "PyTorch-S":
                    tmp += [0, 0, 0]
                else:
                    tmp += [0, 0]
                continue
            results = g[(d, method)]
            tmp += results
        seq_len_glue_data_list.append(tmp)

    for d in datasets2:
        tmp = [d]
        for method in methods:
            if (d, method) not in g:
                if method == "PyTorch-S":
                    tmp += [0, 0, 0]
                else:
                    tmp += [0, 0]
                continue
            results = g[(d, method)]
            tmp += results
        seq_len_long_document_data_list.append(tmp)
    return seq_len_glue_data_list, seq_len_long_document_data_list

def plot_bert(seq_len_glue_data_list, seq_len_long_document_data_list):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    plt.rc('font', size=40) #controls default text size
    plt.rc('axes', titlesize=40) #fontsize of the title
    plt.rc('axes', labelsize=40) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=40) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=40) #fontsize of the y tick labels
    plt.rc('legend', fontsize=40) #fontsize of the legend

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(40, 5))

    keys = ["Pytorch", "Pytorch-S", "Pytorch-S Convert", "DeepSpeed", "TurboTransformer", "PIT"]

    ta = []
    for idx, label in zip(range(2), ["Latency", "GPU Memory"]):

        labels = [ii[0] for ii in seq_len_glue_data_list + seq_len_long_document_data_list]
        if idx == 0:
            pytorch_data = [ii[1] for ii in seq_len_glue_data_list]
            triton_data = [ii[5] for ii in seq_len_glue_data_list]
            triton_convert_data = [ii[7] for ii in seq_len_glue_data_list]
            ours_data = [ii[3] for ii in seq_len_glue_data_list]
            ds_data = [ii[10] for ii in seq_len_glue_data_list]
            turbo_data = [ii[12] for ii in seq_len_glue_data_list]
        else:
            pytorch_data = [ii[2] for ii in seq_len_glue_data_list]
            triton_data = [ii[6] for ii in seq_len_glue_data_list]
            ours_data = [ii[4] for ii in seq_len_glue_data_list]
            ds_data = [ii[11] for ii in seq_len_glue_data_list]
            turbo_data = [ii[13] for ii in seq_len_glue_data_list]
            triton_convert_data = [0 for ii in seq_len_glue_data_list]
    #     print(ours_data[-1])

        x = []
        for i in range(len(labels)):
            x.append(i * 0.5+2.5)
        x = np.array(x)
        xx = x
        width = 0.08
        axs[idx].set_xlim((2.2,x[-1]+0.3))
        # plt.ylim((0, 2000))
        axs[idx].vlines(x[len(seq_len_glue_data_list)] - 0.25, 0.0, 100 if idx == 0 else 7, colors = 'black', linestyles = 'dashed')
        # plt.bar(x-3*width,cublas_50_data, width, label='cuBLAS')
        x, y = x[:len(seq_len_glue_data_list)], x[len(seq_len_glue_data_list):]
        m1 = axs[idx].bar(x-2*width,pytorch_data, width, **styles['torch'])
        m2 = axs[idx].bar(x-1*width,triton_data, width, **styles['torch_s'])

        m3 = axs[idx].bar(x-1*width,triton_convert_data, width, **styles['torch_convert'])
        m4 = axs[idx].bar(x-0*width,ds_data, width, **styles['ds'])
        m5 = axs[idx].bar(x+1*width,turbo_data, width, **styles['turbo'])
        m6 = axs[idx].bar(x+2*width,ours_data, width, **styles['pit'])
        # plt.bar(x-0*width,nmsparse_n1_50_data, width, edgecolor='black', color = 'white', hatch='...', label='nmSPARSE-EW')
        # plt.bar(x+1*width,nmsparse_n4_50_data, width, edgecolor='black', color = 'white', hatch='///', label='nmSPARSE-VW4')
        # plt.bar(x+2*width,nmsparse_n32_50_data, width, edgecolor='black', color = 'white', hatch='oo', label='nmSPARSE-VW32')
        # plt.bar(x+3*width,nmsparse_n4k4_50_data, width, edgecolor='black', color = 'white', hatch='xx', label='nmSPARSE-BW4x4')

        ta.extend([m1, m2, m3, m4, m5, m6])
        if idx == 0:
            axs[idx].set_ylabel('Latency(ms)',)
        else:
            axs[idx].set_ylabel('GPU Memory(GB)',)
        axs[idx].set_xlabel("Datasets",)
    #     axs[idx].set_title(f"{label}",)
        axs[idx].set_xticks(xx)
        axs[idx].set_xticklabels(labels, fontsize=40)

        ax2 = axs[idx].twinx()
        x = y
        if idx == 0:
            pytorch_data = [ii[1] for ii in seq_len_long_document_data_list]
            triton_data = [ii[5] for ii in seq_len_long_document_data_list]
            triton_convert_data = [ii[7] for ii in seq_len_long_document_data_list]
            ours_data = [ii[3] for ii in seq_len_long_document_data_list]
            ds_data = [ii[10] for ii in seq_len_long_document_data_list]
            turbo_data = [ii[12] if ii[12] else 0 for ii in seq_len_long_document_data_list]
        else:
            pytorch_data = [ii[2] for ii in seq_len_long_document_data_list]
            triton_data = [ii[6] for ii in seq_len_long_document_data_list]
            ours_data = [ii[4] for ii in seq_len_long_document_data_list]
            ds_data = [ii[11] for ii in seq_len_long_document_data_list]
            turbo_data = [ii[13] if ii[13] else 0 for ii in seq_len_long_document_data_list]
            triton_convert_data = [0 for ii in seq_len_long_document_data_list]


        ax2.bar(x-2*width,pytorch_data, width,  **styles['torch'])
        ax2.bar(x-1*width,triton_data, width,  **styles['torch_s'])
        ax2.bar(x-1*width,triton_convert_data, width,  **styles['torch_convert'])
        ax2.bar(x-0*width,ds_data, width,  **styles['ds'])
        ax2.bar(x+1*width,turbo_data, width,  **styles['turbo'])
        ax2.bar(x+2*width,ours_data, width,  **styles['pit'])

        for a, b in [(x+1*width,turbo_data), (x-2*width,pytorch_data), (x-1*width,triton_data)]:
        #     print(a, b)
            for ii, jj in zip(a, b):
                if jj == 0:
                    ax2.scatter(ii, jj+(13 if idx == 0 else 0.7), s=100, color="black", marker="x")
    fig.legend(handles=ta, labels=keys, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=6,frameon=False,)
    plt.savefig('figure10.pdf',bbox_inches='tight',pad_inches=0.0,dpi=1000)

    plt.show()


if __name__ == '__main__':
    seq_len_glue_data_list, seq_len_long_document_data_list = load_results()
    plot_bert(seq_len_glue_data_list, seq_len_long_document_data_list)
