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
target_shape = ['M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 
                'M17', 'M18', 'M19', 'M20', 'M21', 'M22', 'M23', 'M24',
                'M25', 'M26', 'M27', 'M28', 'M29', 'M30', 'M31', 'M32']
for f in file_list:
    baseline, shape_id, sparsity = parse_file_name(f)
    f_path = os.path.join(args.prefix, f)
    time_cost = get_kernel_run_time(f_path)
    if time_cost and shape_id in target_shape:
        print(f'SpMM sparsity ratio={sparsity} shape={shape_id} kernel={baseline} latency={time_cost} ')

