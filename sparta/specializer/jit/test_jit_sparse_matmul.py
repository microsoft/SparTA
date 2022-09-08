from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import torch
import numpy as np
import time
from sparta.common import tesa
from sparta import specializer
from sparta.specializer.jit import kernels as jk

# import pycuda.driver as drv
# import pycuda.gpuarray as ga
# from pycuda.compiler import SourceModule

class MatmulDSDTest:

    def __init__(self, cfg: Dict, sparsity: float, transpose: bool, bias: bool) -> None:
        self.cfg = dict(cfg)
        self.transponse = transpose
        self.bias = bias
        self.sparsity = sparsity
        assert sparsity > 0. and sparsity < 1.

        # generate random data
        self.A = torch.rand((cfg['GLOBAL_M_VALUE'], cfg['GLOBAL_K_VALUE']), dtype=torch.float32).cuda()
        self.B = torch.rand((cfg['GLOBAL_N_VALUE'], cfg['GLOBAL_K_VALUE']), dtype=torch.float32).cuda()
        self.B_mask = torch.rand((
            cfg['GLOBAL_N_VALUE'] // cfg['BLOCK_SIZE_N_VALUE'],
            cfg['GLOBAL_K_VALUE'] // cfg['BLOCK_SIZE_K_VALUE'],
        )).cuda() > sparsity
        self.B_mask_upsample = torch.nn.Upsample(scale_factor=(cfg['BLOCK_SIZE_N_VALUE'], cfg['BLOCK_SIZE_K_VALUE']), mode='nearest')(self.B_mask.float().view(1,1,self.B_mask.shape[0],self.B_mask.shape[1])).int().squeeze()
        # B_mask_upsample = np.zeros((cfg['GLOBAL_N_VALUE'], cfg['GLOBAL_K_VALUE']))
        # for row_idx in range(self.B_mask.shape[0]):
        #     for col_idx in range(self.B_mask.shape[1]):
        #         row_start = row_idx * cfg['BLOCK_SIZE_N_VALUE']
        #         row_end = row_start + cfg['BLOCK_SIZE_N_VALUE']
        #         col_start = col_idx * cfg['BLOCK_SIZE_K_VALUE']
        #         col_end = col_start + cfg['BLOCK_SIZE_K_VALUE']
        #         B_mask_upsample[row_start:row_end, col_start:col_end] = self.B_mask[row_idx, col_idx]
        # B_mask_upsample = torch.from_numpy(B_mask_upsample).int()
        # assert torch.equal(B_mask_upsample, B_mask_upsample_2)
        self.B = torch.mul(self.B, self.B_mask_upsample)
        self.Bias = torch.rand((cfg['GLOBAL_N_VALUE'], ), dtype=torch.float32).cuda() if bias else None

        self.GT = torch.mm(self.A, self.B.T)
        if self.Bias is not None:
            self.GT = self.GT + self.Bias
        self.C = torch.empty(self.GT.shape, dtype=torch.float32).cuda()

        B_tesa = tesa.BCSR(
            dense=self.B.cpu().numpy(),
            mask=self.B_mask.cpu().numpy(),
            block_size=(cfg['BLOCK_SIZE_N_VALUE'], cfg['BLOCK_SIZE_K_VALUE'])
        ).sparse
        self.B_val = torch.from_numpy(B_tesa['val']).cuda()
        self.B_ptr = torch.from_numpy(B_tesa['row_ptr']).cuda()
        self.B_idx = torch.from_numpy(B_tesa['col_idx']).cuda()
        
    def test_matmul_kernel(self, kernel: jk.SparseMatMul, repeat: int =1000):
        kernel.matmul(self.A, self.B_val, self.B_ptr, self.B_idx, self.C, self.Bias)
        torch.cuda.synchronize()
        torch.testing.assert_close(self.C,self.GT)

        for _ in range(10):
            kernel.matmul(self.A, self.B_val, self.B_ptr, self.B_idx, self.C, self.Bias)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            kernel.matmul(self.A, self.B_val, self.B_ptr, self.B_idx, self.C, self.Bias)
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end) / repeat
        print(f'PyCuda Latency: {t} ms') 
        return t       

# def prepare_data(cfg):
#     A = np.random.uniform(size=(cfg['GLOBAL_M_VALUE'], cfg['GLOBAL_K_VALUE'])).astype(np.float32)
#     B = np.random.uniform(size=(cfg['GLOBAL_N_VALUE'], cfg['GLOBAL_K_VALUE'])).astype(np.float32)
#     B_mask = np.random.uniform(size=(
#         cfg['GLOBAL_N_VALUE'] // cfg['BLOCK_SIZE_N_VALUE'],
#         cfg['GLOBAL_K_VALUE'] // cfg['BLOCK_SIZE_K_VALUE'],
#     )) < 0.2
#     B_tesa = tesa.BCSR(
#         dense=B,
#         mask=B_mask,
#         block_size=(cfg['BLOCK_SIZE_N_VALUE'], cfg['BLOCK_SIZE_K_VALUE'])
#     ).sparse
#     B_val = B_tesa['val']
#     bias = np.random.uniform(size=(cfg['GLOBAL_N_VALUE'], )).astype(np.float32)

#     B_mask_tiled = np.zeros((cfg['GLOBAL_N_VALUE'], cfg['GLOBAL_K_VALUE']))
#     for row_idx in range(B_mask.shape[0]):
#         for col_idx in range(B_mask.shape[1]):
#             row_start = row_idx * cfg['BLOCK_SIZE_N_VALUE']
#             row_end = row_start + cfg['BLOCK_SIZE_N_VALUE']
#             col_start = col_idx * cfg['BLOCK_SIZE_K_VALUE']
#             col_end = col_start + cfg['BLOCK_SIZE_K_VALUE']
#             B_mask_tiled[row_start:row_end, col_start:col_end] = B_mask[row_idx, col_idx]

#     B *= B_mask_tiled
#     C_tgt = A @ B.T 
#     return A, B, B_tesa, B_mask, bias, C_tgt


cfg = {
    'GLOBAL_M_VALUE': 1024,
    'GLOBAL_K_VALUE': 256,
    'GLOBAL_N_VALUE': 512,
    'BLOCK_SIZE_M_VALUE': 64,
    'BLOCK_SIZE_K_VALUE': 8,
    'BLOCK_SIZE_N_VALUE': 128,
    'THREAD_SIZE_M_VALUE': 8,
    'THREAD_SIZE_K_VALUE': 4,
    'THREAD_SIZE_N_VALUE': 16
}

bias = True
test = MatmulDSDTest(cfg, 0.8, transpose=True, bias=bias)
s = jk.SparseMatMul('dsd', transpose=True, bias=bias)
s.set_parameters(cfg)
s.compile()
test.test_matmul_kernel(s)

# A, B, B_tesa, B_mask, bias, C_tgt = prepare_data(cfg)
# TG = torch.from_numpy(C_tgt).cuda()
# AG = torch.from_numpy(A).cuda()
# BG = {k:torch.from_numpy(v).cuda() for k,v in B_tesa.items()}
# VALG = torch.from_numpy(B_tesa['val']).cuda()
# PTRG = torch.from_numpy(B_tesa['row_ptr']).cuda()
# IDXG = torch.from_numpy(B_tesa['col_idx']).cuda()
# CG = torch.empty(C_tgt.shape).cuda()


