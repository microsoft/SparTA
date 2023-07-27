import re
import numpy as np
import matplotlib.pyplot as plt


SPARSITY_LIST = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def read_latency_from_file(path):
    with open(path) as f:
        log = f.readlines()
    for line in log:
        if 'latency' in line.lower() or 'time' in line.lower():
            return float(re.findall('\d+\.\d+', line)[0])


def read_data(prefix):
    return [
        read_latency_from_file(f'./log/{prefix}_{sparsity}.log')
        for sparsity in SPARSITY_LIST
    ]


plt.rc('font', size=25) #controls default text size
plt.rc('axes', titlesize=25) #fontsize of the title
plt.rc('axes', labelsize=25) #fontsize of the x and y labels
plt.rc('xtick', labelsize=25) #fontsize of the x tick labels
plt.rc('ytick', labelsize=25) #fontsize of the y tick labels
plt.rc('legend', fontsize=25) #fontsize of the legend
plt.figure(figsize=(8,3))

labels = [int(x * 100) for x in SPARSITY_LIST]

ori_data = read_data('ori')
pit_data = read_data('pit')

x = []
for i in range(len(labels)):
    x.append(i * 0.3+2.4)
x = np.array(x)
width = 0.1
plt.xlim((2.2,x[-1]+0.2))
plt.ylim((0, 10))

plt.bar(x-0.5*width, pit_data, width, edgecolor='black', color = 'white', label='32x1')
plt.bar(x+0.5*width, ori_data, width, edgecolor='black', color = 'white', hatch='\\\\\\', label='32x64')
# plt.bar(x-0*width,nmsparse_n1_50_data, width, edgecolor='black', color = 'white', hatch='...', label='nmSPARSE-EW')
# plt.bar(x+1*width,nmsparse_n4_50_data, width, edgecolor='black', color = 'white', hatch='///', label='nmSPARSE-VW4')
# plt.bar(x+2*width,nmsparse_n32_50_data, width, edgecolor='black', color = 'white', hatch='oo', label='nmSPARSE-VW32')
# plt.bar(x+3*width,nmsparse_n4k4_50_data, width, edgecolor='black', color = 'white', hatch='xx', label='nmSPARSE-BW4x4')
            
plt.ylabel('Latency(ms)')
plt.xlabel("Sparsity(%)")

# plt.title()
plt.xticks(x, labels=labels)
plt.yticks([0, 2.5, 5, 7.5, 10], labels=[0, 2.5, 5, 7.5, 10])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4,frameon=False)
plt.savefig('figure16.pdf',bbox_inches='tight',pad_inches=0.0,dpi=1000)
