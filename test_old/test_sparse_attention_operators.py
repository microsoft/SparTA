# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
import warnings

import torch

import sparta
from sparta.testing import block_mask


B = 4
H = 4
E = 256
Nt = 3072
Ns = 768
SPARSITY = 0.99
BLOCK = (8, 8)
COMMON_TILE_CONFIG = {
    'BH': 64,
    'BW': 64,
}
MATMUL_IN_TILE_CONFIG = {
    'MATMUL_IN_BK': 32,
    'MATMUL_IN_TM': 4,
    'MATMUL_IN_TK': 4,
    'MATMUL_IN_TN': 4,
}
SOFTMAX_TILE_CONFIG = {
    'SOFTMAX_RT': 4,
}
MATMUL_OUT_TILE_CONFIG = {
    'MATMUL_OUT_BN': 32,
    'MATMUL_OUT_TM': 4,
    'MATMUL_OUT_TK': 4,
    'MATMUL_OUT_TN': 4,
}


def test_attention_operator(
    matmul_in_impl: str, softmax_impl: str, matmul_out_impl: str,
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor
):
    sparse_attention = sparta.nn.SparseAttention(B, Ns, Nt, H, E, mask)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dense_out = sparse_attention.forward(Q, K, V)
    print(f'----- {matmul_in_impl} MatMul, {softmax_impl} Softmax, {matmul_out_impl} MatMul -----')
    sparse_attention.build(dict(
        _name=f'{matmul_in_impl.lower()}_{softmax_impl.lower()}_{matmul_out_impl.lower()}',
        **COMMON_TILE_CONFIG,
        **MATMUL_IN_TILE_CONFIG,
        **SOFTMAX_TILE_CONFIG,
        **MATMUL_OUT_TILE_CONFIG,
    ), sample_inputs=[Q, K, V])
    sparse_out = sparse_attention.forward(Q, K, V)
    torch.testing.assert_close(dense_out, sparse_out, rtol=1e-4, atol=1e-4)
    print('PASS')


class TestSparseAttentionOperators(unittest.TestCase):

    def test_sparse_linear(self):
        print('==================== Testing Sparse Attention Operators ====================')
        torch.manual_seed(2022)
        Q = torch.rand((B, H, Nt, E), dtype=torch.float32).cuda()
        K = torch.rand((B, H, Ns, E), dtype=torch.float32).cuda()
        V = torch.rand((B, H, Ns, E), dtype=torch.float32).cuda()
        mask = block_mask((Nt, Ns), block=BLOCK, sparsity=SPARSITY).cuda()
        test_attention_operator('SparTA', 'SparTA', 'SparTA', Q, K, V, mask)
        test_attention_operator('OpenAI', 'SparTA', 'SparTA', Q, K, V, mask)
        test_attention_operator('SparTA', 'SparTA', 'OpenAI', Q, K, V, mask)


if __name__ == '__main__':
    unittest.main()
