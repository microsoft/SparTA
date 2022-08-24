import os
import time

import torch
import numpy as np

os.sys.path.append(os.getcwd())

from sparta.common import tesa
from sparta import specializer

np.random.seed(2022)

cfg = {
    'GLOBAL_M_VALUE': 1024,
    'GLOBAL_K_VALUE': 256,
    'GLOBAL_N_VALUE': 512,
}

factory = specializer.get_factory('sparse_linear_openai_dds_t')

# Prepare Data
def prepare_data():
    A = np.random.uniform(size=(cfg['GLOBAL_M_VALUE'], cfg['GLOBAL_K_VALUE'])).astype(np.float32)
    B = np.random.uniform(size=(cfg['GLOBAL_N_VALUE'], cfg['GLOBAL_K_VALUE'])).astype(np.float32)

    C_tgt = (A @ B.T).astype(np.float32)
    C_mask = np.random.uniform(size=(
        cfg['GLOBAL_M_VALUE'] // 32,
        cfg['GLOBAL_N_VALUE'] // 32,
    )) < 0.2
    C_tesa = tesa.BCSR(
        dense=C_tgt,
        mask=C_mask,
        block_size=(32, 32),
        mode='X',
    ).sparse
    C_val = C_tesa['val']

    return A, B, C_val, C_mask, C_tgt

A, B, C_val, C_mask, C_tgt = prepare_data()

# Test Function
test_func = factory.get_test_interface(cfg, mask={'C': C_mask})
print(f'NVCC Latency: {test_func(inputs={"A": A, "B": B}, num_iters=1000)} ms')

# PyTorch Module
module_interface = factory.get_module_interface(cfg, mask={'C': C_mask})
module_code = module_interface.get_module_code()
with open('./test/module.cu', 'w') as f:
    f.write(module_code)

f = module_interface.get_module().forward

device = torch.device(f'cuda:3')
A = torch.from_numpy(A).to(device)
B = torch.from_numpy(B).to(device)

for _ in range(10):
    C = f(A, B)
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    C = f(A, B)
torch.cuda.synchronize()
print(f'PyTorch Latency: {(time.time() - start)} ms')

C = C.cpu().numpy()
print(f'Sum_C: {C.sum()}')
print(f'Sum_C_tgt: {C_val.sum()}')
print(f'Error: {np.sum(np.abs(C - C_val))}')
