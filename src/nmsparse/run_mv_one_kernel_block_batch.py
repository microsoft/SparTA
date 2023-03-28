import os
from typing import Dict, List, Optional, Tuple
import subprocess
import json
from pathlib import Path

current_path = Path(__file__).parent

def kernel_execution(kernel: str) -> Tuple[str, float, bool]:
    # kernel execution process
    file_name_new = "kernel_generate_code.cu"
    new_file_path = os.path.join(current_path, "build", file_name_new)
    if not os.path.exists(os.path.join(current_path, "build")):
        os.makedirs(os.path.join(current_path, "build")) 
    with open(new_file_path, 'w') as f:
        f.write(kernel)
    avg_latency, success = run_gpu_kernel(file_name_new)
    # kernel correctness verification failure
    if success == False:
        avg_latency = 10000
    return kernel, avg_latency, success

def run_gpu_kernel(file_name):
    file_path = os.path.join(current_path, "build", file_name)
    executor_path = os.path.splitext(file_path)[0]
    compile_cmd = 'nvcc -gencode arch=compute_80,code=sm_80 \
    {} -o {}'.format(file_path, executor_path)
    output_file_name = f"output_log.txt"
    output_file_path = os.path.join(current_path, "build", output_file_name)
    subprocess.check_output(compile_cmd, shell = True, universal_newlines=True, timeout=600)
    latencys = []
    for i in range(2):
        command = '{} > {}'.format(executor_path, output_file_path)
        #os.system('nvprof --unified-memory-profiling off ./{} 2> a_{}.txt'.format(Path(file_name).stem, file_name))
        #os.system(command)
        subprocess.check_output(command, shell = True, universal_newlines=True, timeout=600)

        if i == 0:
            continue
        latencys.append(get_kernel_run_time('{}'.format(output_file_path)))
    success = verify_successful(output_file_path)
    avg_latency = sum(latencys) / len(latencys)
    return avg_latency, success


def get_kernel_run_time(file_name):
    lines = []
    kernel_name = "Time="
    with open(file_name, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.find(kernel_name) == -1:
            continue
        else:
            run_time = float(line.split()[-2])
            break
    return run_time

def verify_successful(file_name):
    with open(file_name, 'r') as f:
        content = f.read()
    if content.find("Pass") == -1:
        return False
    return True

def run_kernel(config):
    template_name = os.path.join(current_path, "template","MV_one_kernel_block_batch.cu")
    f_template = open(template_name)
    template_str = f_template.read()
    for key, value in config.items():
        template_str = template_str.replace(key, str(value))
    kernel, avg_latency, success = kernel_execution(template_str)

    M = config['M_GLOBAL_VAL']
    K = config['K_GLOBAL_VAL']
    N = config['N_GLOBAL_VAL']
    sparsity = config['SPARSITY_RATIO_VAL']
    print(f"M:{M}, K:{K}, N:{N}, sparsity:{sparsity}, success:{success}, time:{avg_latency}")

def main():
    test_cases_spmv_b1 = [[1, 1024, 1024], [1, 2048, 2048], [1, 4096, 4096],\
    [1, 8192, 8192], [1, 1024, 4096], [1, 4096, 1024], [1,5120,20480], [1,20480,5120]]
    # ratios =  [0.5, 0.75, 0.90625]
    ratios =  [0.875]
    for i in range(len(test_cases_spmv_b1)):
        test_case = test_cases_spmv_b1[i]
        for sparsity in ratios:
            config = {}
            config['M_GLOBAL_VAL'] = test_case[0]
            config['K_GLOBAL_VAL'] = test_case[1]
            config['N_GLOBAL_VAL'] = test_case[2]
            config['SPARSITY_RATIO_VAL'] = sparsity
            run_kernel(config)

main()
