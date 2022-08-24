import os

import torch
import numpy as np

os.sys.path.append(os.getcwd())

import sparta

np.random.seed(2022)
torch.manual_seed(2022)

space = {
    'BLOCK_SIZE_M_VALUE': [32, 64],
    'BLOCK_SIZE_K_VALUE': [32, 64],
    'BLOCK_SIZE_N_VALUE': [32, 64],
    # 'THREAD_SIZE_M_VALUE': [2, 4],
    # 'THREAD_SIZE_K_VALUE': [2, 4],
    # 'THREAD_SIZE_N_VALUE': [2, 4],
    'THREAD_SIZE_M_VALUE': [4],
    'THREAD_SIZE_K_VALUE': [4],
    'THREAD_SIZE_N_VALUE': [4],
}

M, K, N = 4096, 3072, 768

B_mask = np.random.uniform(size=(K, N)) < 0.001
dense_linear = torch.nn.Linear(K, N)
best_config = sparta.find_best_config(dense_linear, mask=B_mask, search_space=space)
our_linear = sparta.SparseLinear(dense_linear, B_mask, best_config)
C = our_linear.forward(torch.rand((M, K), dtype=torch.float32))
print(C.sum().item())
