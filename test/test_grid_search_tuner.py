import os

import torch

os.sys.path.append(os.getcwd())

from sparta.specializer import operators

torch.manual_seed(2022)
device = torch.device('cuda')

space = {
    'sparta': {
        'BLOCK_SIZE_M_VALUE': [32, 64],
        'BLOCK_SIZE_K_VALUE': [32, 64],
        'BLOCK_SIZE_N_VALUE': [32, 64],
        'THREAD_SIZE_M_VALUE': [2, 4],
        'THREAD_SIZE_K_VALUE': [2, 4],
        'THREAD_SIZE_N_VALUE': [2, 4],
    }
}

M, K, N = 1024, 1024, 1024

A = torch.rand((M, K)).to(device)
B_mask = torch.rand((K, N)) > 0.999
dense_linear = torch.nn.Linear(K, N)

sparse_linear = operators.SparseLinear(dense_linear, weight_mask=B_mask)
impl, config = sparse_linear.tune(sample_inputs=[A], search_space=space)

sparse_linear.build(impl, config)
sparse_linear.to(device)
C = sparse_linear.forward(A)
print(f'Sum(C) = {C.sum()}')
