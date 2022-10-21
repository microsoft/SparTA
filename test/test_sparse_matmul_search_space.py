# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
import json

from sparta.specializer.funtional import SparseBatchMatMulCtx


# BATCH_SIZE, M, K, N = 4, 1024, 256, 512
# BM, BK, BN, TM, TK, TN = 32, 32, 32, 4, 4, 4


def test_sparse_matmul_search_space(
    sparse_type: str, biased: bool, compressed: bool, trans_A: bool, trans_B: bool
):
    b_str = '_b' if biased else ''
    c_str = '_c' if compressed else ''
    t_A_str = 't' if trans_A else 'n'
    t_B_str = 't' if trans_B else 'n'
    func_name = f'sparse_matmul_{sparse_type}{b_str}_{t_A_str}{t_B_str}{c_str}'
    print(func_name)

    sparse_ctx = SparseBatchMatMulCtx(sparse_type, trans_A, trans_B, biased, compressed)
    search_space = sparse_ctx.get_search_space(backward=True)
    # print(sparse_ctx.get_search_space(backward=True))
    with open('tmp.json', 'w') as f:
        f.write(json.dumps({
            impl: {
                '_space': {k: str(v._value) for k, v in space['_space'].items()},
                '_conditions': space['_conditions'],
            }
            for impl, space in search_space.items()
        }, indent='\t'))


class TestSparseMatMulSearchSpace(unittest.TestCase):

    def test_sparse_matmul_search_spaces(self):
        print('==================== Testing Sparse MatMul Search Space ====================')
        # for sparse_type in ['sdd', 'dsd', 'dds']:
        #     for biased in [False, True]:
        #         for trans_A in [False, True]:
        #             for trans_B in [False, True]:
        #                 for compressed in [False, True]:
        #                     test_sparse_matmul_search_space(
        #                         sparse_type, biased, compressed, trans_A, trans_B
        #                     )
        test_sparse_matmul_search_space('sdd', False, True, False, False)


if __name__ == '__main__':
    unittest.main()
