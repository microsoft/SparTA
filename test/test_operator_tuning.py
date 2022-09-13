# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch

import sparta
from sparta.common.tuning import TunableItemCfg, Tunable


M, K, N = 4096, 3072, 768
SEARCH_SPACE = TunableItemCfg('choice', _is_nested=True, _value={
    'openai': {},
    'sparta': {
        'BLOCK_SIZE_M_VALUE': TunableItemCfg('choice', [32, 64]),
        'BLOCK_SIZE_K_VALUE': TunableItemCfg('choice', [32, 64]),
        'BLOCK_SIZE_N_VALUE': TunableItemCfg('choice', [32, 64]),
        'THREAD_SIZE_M_VALUE': TunableItemCfg('choice', [4]),
        'THREAD_SIZE_K_VALUE': TunableItemCfg('choice', [4]),
        'THREAD_SIZE_N_VALUE': TunableItemCfg('choice', [4]),
    },
})


class TestOperatorTuning(unittest.TestCase):

    def _test_parse_space(self, cfg, space, samples=None):
        t = Tunable(cfg, 'test')
        self.assertDictEqual(t.search_space, space)
        tuner = t.create_tuner('grid')
        if samples:
            for i, sample in enumerate(samples):
                ts = tuner.generate_parameters(i)
                self.assertDictEqual(ts, sample)

    def test_search_space_parsing(self):
        # simple search space
        cfg = TunableItemCfg('choice', [32, 64])
        ss = {
            'test': {
                '_type': 'choice',
                '_value': [32, 64],
            }
        }
        samples = [{'test': 32}, {'test': 64}]
        self._test_parse_space(cfg, ss, samples)
        # simple search space
        cfg = TunableItemCfg('choice', _is_nested=True, _value={
            'sparta': {}
        })
        ss = {
            'test': {
                '_type': 'choice',
                '_value': [{'_name': 'sparta'}],
            }
        }
        samples = [{'test': {'_name': 'sparta'}}]
        self._test_parse_space(cfg, ss, samples)
        # nested search space
        cfg = TunableItemCfg('choice', _is_nested=True, _value={
            'openai': {},
            'sparta': {
                'BM': TunableItemCfg('choice', [32, 64]),
                'BN': TunableItemCfg('choice', [8, 16]),
            }
        })
        ss = {
            'test': {
                '_type': 'choice',
                '_value': [
                    {'_name': 'openai'},
                    {
                        '_name': 'sparta',
                        'BM': {'_type': 'choice', '_value': [32, 64]},
                        'BN': {'_type': 'choice', '_value': [8, 16]},
                    },
                ],
            }
        }
        samples = [
            {'test': {'_name': 'openai'}},
            {'test': {'BM': 32, 'BN': 8, '_name': 'sparta'}},
        ]
        self._test_parse_space(cfg, ss, samples)

    def test_tune_sparse_linear_dsd(self):
        print('==================== Testing Grid Search Tuner ====================')
        dense_input = torch.rand((M, K)).cuda()
        weight = torch.rand((N, K)).cuda()
        weight_mask = sparta.testing.block_mask(shape=(N, K)).cuda()
        weight = torch.mul(weight, weight_mask)
        dense_op = torch.nn.Linear(K, N, bias=False).cuda()
        dense_op.load_state_dict(dict(weight=weight))
        sparse_op = sparta.nn.SparseLinear(dense_op, weight_mask=weight_mask)
        # test default search space
        sparse_op.set_search_space(SEARCH_SPACE)
        best_params = sparse_op.tune(
            sample_inputs=[dense_input],
            algo='grid'
        )
        print(f'Best params: {best_params}')
        sparse_op.build(best_params, sample_inputs=[dense_input])
        torch.testing.assert_close(sparse_op(dense_input), dense_op(dense_input))
        print(f'PASS')


if __name__ == '__main__':
    unittest.main()
