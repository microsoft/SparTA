# torch.utils.cpp_extension.load

import os
import time
import shutil
from jinja2 import Template
import pycuda.autoinit

tmp_dir = 'tmp'
# shutil.rmtree(tmp_dir, ignore_errors=True)
# os.makedirs(tmp_dir, exist_ok=True)

# templates_dir = 'SparTA/Specializer/Templates'
# dependencies = ['sparse_matmul.cuh']
# for file_name in dependencies:
#     source_path = os.path.join(templates_dir, 'Kernels', file_name)
#     target_path = os.path.join(tmp_dir, file_name)
#     shutil.copyfile(source_path, target_path)

# operator_name = 'sparse_linear'
# with open(os.path.join(templates_dir, 'Operators', f'{operator_name}.cu.j2')) as f:
#     operator_template = f.read()
# operator_code = Template(operator_template).render({
#     'BIASED': True,
#     'TYPE': 'float',
#     'GLOBAL_M_VALUE': 1024,
#     'GLOBAL_K_VALUE': 1024,
#     'GLOBAL_N_VALUE': 1024,
#     'BLOCK_SIZE_M_VALUE': 64,
#     'BLOCK_SIZE_K_VALUE': 8,
#     'BLOCK_SIZE_N_VALUE': 128,
#     'THREAD_SIZE_M_VALUE': 8,
#     'THREAD_SIZE_K_VALUE': 4,
#     'THREAD_SIZE_N_VALUE': 8
# })
# with open(os.path.join(tmp_dir, f'{operator_name}.cu'), 'w') as f:
#     f.write(operator_code)

print('Load kernel ...')
start = time.time()
# from torch.utils.cpp_extension import load
# sp_linear_function = load(
#     name="sp_linear",
#     sources=['./tmp/sparse_linear.cu', './tmp/sparse_matmul.cuh'],
#     extra_cflags=["-std=c++14", "-O3"],
#     extra_cuda_cflags=["--x=cu"],
#     # extra_include_paths=['./tmp'], 52.43022108078003 s 51.63953614234924 s
#     with_cuda=True,
#     verbose=True
# )

# nvcc -gencode arch=compute_61,code=sm_61 ./tmp/sparse_matmul.cu -o ./tmp/block_sparse

# template <typename T, int M, int K, int N, int BM, int BK, int BN, int TM, int TK, int TN>
# __global__ void BLOCK_SPARSE_MATMUL_BIASED(
#     T *__restrict__ input_A,
#     T *__restrict__ input_W_val,
#     int* input_W_row,
#     int* input_W_col,
#     T *__restrict__ input_bias,
#     T *__restrict__ output_C
# )

from pycuda.compiler import SourceModule
with open(os.path.join(tmp_dir, 'sparse_matmul.cu')) as f:
    mod = SourceModule(f.read(), arch='compute_61', code='sm_61')

sp_linear_function = mod.get_function("sparse_linear")

print(f'Finished in {time.time() - start} s')