import os

import torch
import numpy as np

os.sys.path.append(os.getcwd())

from sparta.specializer import operators

np.random.seed(2022)
torch.manual_seed(2022)
device = torch.device('cuda:0')

space = {
    'our': {
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
B_mask = np.random.uniform(size=(K, N)) < 0.001
dense_linear = torch.nn.Linear(K, N)

sparse_linear = operators.SparseLinear(dense_linear, weight_mask=B_mask)
best_config = sparse_linear.tune(sample_inputs=[A], search_space=space)

sparse_linear.build(best_config)
sparse_linear.to(device)
C = sparse_linear.forward(A)
