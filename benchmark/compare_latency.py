import unittest
from typing import Tuple

import torch
import triton
import pandas as pd

from sparta.nn import SparseLinear
from sparta.nn.module_tuner import tune_sparse_operator
from sparta.testing import block_mask, test_latency


M, K, N = 1024, 1024, 1024
BLOCK_SIZE_ARR = [1, 8, 32]
SPARSITY_ARR = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]

device = torch.device(f'cuda:0')

def prepare_data(block: Tuple[int, int], sparsity: float, seed: int = 2022):
    torch.manual_seed(seed)
    A = torch.rand((M, K), dtype=torch.float32).cuda()
    B = torch.rand((N, K), dtype=torch.float32).cuda()
    mask = block_mask((N, K), block, sparsity).cuda()
    B *= mask
    return A, B, mask

def test_triton(A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor, block_size: int = 32):
    A_tri = A.reshape((1, 1) + A.shape)
    B_tri = B.reshape((1, 1) + B.shape)

    layout = mask.reshape(N // block_size, block_size, K // block_size, block_size)
    layout = layout.swapaxes(1, 2).all(-1).all(-1).unsqueeze(0).to(torch.int32)
    print(layout.sum(), layout.sum() / N / K * block_size * block_size)

    B_tri = triton.testing.sparsify_tensor(B_tri, layout, block_size)

    # A_tri = A_tri.cuda()
    # B_tri = B_tri.cuda()
    # layout = layout.cuda()
    tri_matmul = triton.ops.blocksparse.matmul(
        layout=layout,
        block=block_size,
        mode='dds',
        device='cuda:0',
        trans_a=False,
        trans_b=True,
    )

    # triton.testing.do_bench(lambda: op(A_tri, B_tri), warmup=1000, rep=1000)
    return test_latency(tri_matmul, inputs=[A_tri, B_tri]), test_latency(tri_matmul, inputs=[A_tri, B_tri], cuda=True)


def test_sparta(A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor):
    dense_linear = torch.nn.Linear(K, N, bias=False).cuda()
    dense_linear.load_state_dict({'weight': B})
    sparse_linear = SparseLinear(dense_linear, weight_mask=mask)
    tune_sparse_operator(sparse_linear, sample_inputs=[A], max_trials=1, algo='rand')
    return test_latency(sparse_linear, inputs=[A]), test_latency(sparse_linear, inputs=[A], cuda=True)


def test_dense(A: torch.Tensor, B: torch.Tensor):
    dense_linear = torch.nn.Linear(K, N, bias=False).cuda()
    dense_linear.load_state_dict({'weight': B})
    return test_latency(dense_linear, inputs=[A]), test_latency(dense_linear, inputs=[A], cuda=True)


class TestLatency(unittest.TestCase):

    def test_sparta_linear_latency(self):
        print('==================== Test SparTA Sparse Linear Latency ====================')
        data = []
        columns = ['METHOD', 'M', 'K', 'N', 'BLOCK_SIZE', 'SPARSITY', 'LATENCY']
        for block_size in BLOCK_SIZE_ARR[-1:]:
            for sparsity in SPARSITY_ARR:
                A, B, mask = prepare_data((block_size, block_size), sparsity)
                if mask.sum() == 0:
                    continue
                latency, cuda_latency = test_sparta(A, B, mask)
                data.append(['sparta', M, K, N, block_size, sparsity, latency])
                data.append(['sparta_cuda', M, K, N, block_size, sparsity, cuda_latency])
                print(f'block_size={block_size}, sparsity={sparsity} => {cuda_latency} / {latency} ms')
            pd.DataFrame(data, columns=columns).to_csv('test/tmp/latency/sparta.csv', index=False)

    def test_triton_linear_latency(self):
        print('==================== Test Triton Sparse MatMul Latency ====================')
        data = []
        columns = ['METHOD', 'M', 'K', 'N', 'BLOCK_SIZE', 'SPARSITY', 'LATENCY']
        for tri_block in [16, 32, 64]:  
            for block_size in BLOCK_SIZE_ARR:
                for sparsity in SPARSITY_ARR:
                    A, B, mask = prepare_data((block_size, block_size), sparsity)
                    if mask.sum() == 0:
                        continue
                    latency, cuda_latency = test_triton(A, B, mask, tri_block)
                    data.append([f'tri_{tri_block}', M, K, N, block_size, sparsity, latency])
                    data.append([f'tri_{tri_block}_cuda', M, K, N, block_size, sparsity, cuda_latency])
                    print(f'block_size={block_size}/{tri_block}, sparsity={sparsity} => {cuda_latency} / {latency} ms')
        pd.DataFrame(data, columns=columns).to_csv('test/tmp/latency/triton.csv', index=False)

    def test_dense_linear_latency(self):
        print('==================== Test PyTorch Dense Linear Latency ====================')
        data = []
        columns = ['METHOD', 'M', 'K', 'N', 'BLOCK_SIZE', 'SPARSITY', 'LATENCY']
        for block_size in BLOCK_SIZE_ARR:
            for sparsity in SPARSITY_ARR:
                A, B, mask = prepare_data((block_size, block_size), sparsity)
                if mask.sum() == 0:
                    continue
                latency, cuda_latency = test_dense(A, B)
                data.append(['dense', M, K, N, block_size, sparsity, latency])
                data.append([f'dense_cuda', M, K, N, block_size, sparsity, cuda_latency])
                print(f'block_size={block_size}, sparsity={sparsity} => {cuda_latency} / {latency} ms')
        pd.DataFrame(data, columns=columns).to_csv('test/tmp/latency/dense.csv', index=False)

if __name__ == '__main__':
    unittest.main()
