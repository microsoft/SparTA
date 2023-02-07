import re
import os
import sys
import csv

#sparsity_ratio=(0.5, 0.75, 0.9)
sparsity_ratio=(0.5,)
M=(1, 16, 256 ,1024, 4096)
#KN=(1024, 2048, 4096, 8192)
KN=((1024, 1024),(2048, 2048),(4096,4096),(8192,8192),  (1024, 4096), (4096, 1024), (5120, 20480), (20480, 5120))
#baseline = ['sputnik', 'cusparse']
baseline = ['cusparselt']
result = []
for s in sparsity_ratio:
    for b in baseline:
        for kn in KN:
            k, n = kn
            for m in M:
                fpath = './log/{}_{}_{}_{}_{}.log'.format(b, m, k, n, s)
                if not os.path.exists(fpath):
                    continue
                with open(fpath) as f:
                    lines = f.readlines()
                    lines = [line for line in lines if 'Time=' in line]
                    if len(lines) == 0:
                        continue
                    tmp = re.split(' ', lines[0])
                    time = float(tmp[1])
                    print(s, m, k, n, b, time)
                    result.append((s, m, k, n, b, time))
with open('baseline.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    for row in result:
        writer.writerow([str(v) for v in row])

