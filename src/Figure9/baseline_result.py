import sys
import os
import argparse
import re
def get_kernel_run_time(file_name):
    lines = []
    kernel_name = "Time="
    with open(file_name, 'r') as f:
        lines = f.readlines()
    run_time =None
    for line in lines:
        if line.find(kernel_name) == -1:
            continue
        else:
            run_time = float(line.split()[-2])
            break
    return run_time

def parse_file_name(fname):
    return re.split('_', fname[:-3])

parser = argparse.ArgumentParser()
parser.add_argument('--prefix')
args = parser.parse_args()
file_list = os.listdir(args.prefix)
target_shape = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
for f in file_list:
    baseline, shape_id, sparsity = parse_file_name(f)
    f_path = os.path.join(args.prefix, f)
    time_cost = get_kernel_run_time(f_path)
    if time_cost and shape_id in target_shape:
        print(f'SpMV sparsity ratio={sparsity} shape={shape_id} kernel={baseline} latency={time_cost} ')

