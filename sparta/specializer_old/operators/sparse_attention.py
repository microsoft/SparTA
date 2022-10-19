# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import warnings
from typing import Callable, Dict, List

import torch
import numpy as np

from sparta.specializer import kernels
from sparta.common.tuning import TunableItemCfg
from sparta.specializer.operators.operator_base import OperatorBase
from sparta.testing import test_latency, sparse_softmax_reference


class SparseAttention(OperatorBase):

    def __init__(
        self, batch_size: int, src_seq_len: int, tgt_seq_len: int, num_heads: int, embed_dim: int,
        mask: torch.Tensor, dropout: float = 0.0, dtype: str = 'float'
    ):
        assert mask.shape == (tgt_seq_len, src_seq_len), 'Invalid mask shape'
        assert embed_dim % num_heads == 0, 'Embed dims must be divided by number of heads'
        super(OperatorBase, self).__init__()  # TODO: refactor OperatorBase to adapt SparseAttention
        self._mask_cuda = mask.cuda().to(torch.int32)
        self._mask: np.ndarray = mask.cpu().detach().numpy()
        self._batch_size = batch_size
        self._num_heads = num_heads
        self._tgt_seq_len = tgt_seq_len
        self._src_seq_len = src_seq_len
        self._embed_dim = embed_dim
        self._dropout = dropout

        self._matmul_in_possible_implementations: Dict[str, kernels.MatMulKernelBase] = {
            'sparta': kernels.SparTATemplateSparseMatMulKernel(
                sparse_type='dds',
                batch_size=batch_size * num_heads,
                dtype=dtype,
                biased=False,
                transpose=True,
                compressed=True,
            ),
            'openai': kernels.OpenAITemplateSparseMatMulKernel(
                sparse_type='dds',
                batch_size=batch_size * num_heads,
                dtype=dtype,
                biased=False,
                transpose=True,
                compressed=True,
            ),
        }
        self._softmax_possible_implementations: Dict[str, kernels.SoftmaxKernelBase] = {
            'sparta': kernels.SparTATemplateSparseSoftmaxKernel(
                batch_size=batch_size * num_heads,
                dtype=dtype,
                compressed=True,
            ),
        }
        self._matmul_out_possible_implementations: Dict[str, kernels.MatMulKernelBase] = {
            'sparta': kernels.SparTATemplateSparseMatMulKernel(
                sparse_type='sdd',
                batch_size=batch_size * num_heads,
                dtype=dtype,
                biased=False,
                transpose=False,
                compressed=True,
            ),
            'openai': kernels.OpenAITemplateSparseMatMulKernel(
                sparse_type='sdd',
                batch_size=batch_size * num_heads,
                dtype=dtype,
                biased=False,
                transpose=False,
                compressed=True,
            ),
        }
        
        self._matmul_in_forward_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None
        self._softmax_forward_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None
        self._matmul_out_forward_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None
        self._search_space = None
        self.ready = False

    def _read_params(self, params: Dict):
        matmul_in_impl, softmax_impl, matmul_out_impl = params['_name'].split('_')
        matmul_in_params = {'_name': matmul_in_impl}
        softmax_params = {'_name': softmax_impl}
        matmul_out_params = {'_name': matmul_out_impl}
        if matmul_in_impl == 'sparta':
            matmul_in_params['BLOCK_SIZE_M_VALUE'] = params['BH'] if matmul_out_impl == 'sparta' else 32
            matmul_in_params['BLOCK_SIZE_K_VALUE'] = params['MATMUL_IN_BK']
            matmul_in_params['BLOCK_SIZE_N_VALUE'] = params['BW'] if matmul_out_impl == 'sparta' else 64
            matmul_in_params['THREAD_SIZE_M_VALUE'] = params['MATMUL_IN_TM']
            matmul_in_params['THREAD_SIZE_K_VALUE'] = params['MATMUL_IN_TK']
            matmul_in_params['THREAD_SIZE_N_VALUE'] = params['MATMUL_IN_TN']
            softmax_params['BLOCK_SIZE_H_VALUE'] = matmul_in_params['BLOCK_SIZE_M_VALUE']
            softmax_params['BLOCK_SIZE_W_VALUE'] = matmul_in_params['BLOCK_SIZE_N_VALUE']
        if matmul_out_impl == 'sparta':
            matmul_out_params['BLOCK_SIZE_M_VALUE'] = params['BH'] if matmul_in_impl == 'sparta' else 32
            matmul_out_params['BLOCK_SIZE_K_VALUE'] = params['BW'] if matmul_in_impl == 'sparta' else 32
            matmul_out_params['BLOCK_SIZE_N_VALUE'] = params['MATMUL_OUT_BN']
            softmax_params['BLOCK_SIZE_H_VALUE'] = matmul_out_params['BLOCK_SIZE_M_VALUE']
            softmax_params['BLOCK_SIZE_W_VALUE'] = matmul_out_params['BLOCK_SIZE_K_VALUE']
            matmul_out_params['THREAD_SIZE_M_VALUE'] = params['MATMUL_OUT_TM']
            matmul_out_params['THREAD_SIZE_K_VALUE'] = params['MATMUL_OUT_TK']
            matmul_out_params['THREAD_SIZE_N_VALUE'] = params['MATMUL_OUT_TN']
        softmax_params['ROW_TILE_VALUE'] = params['SOFTMAX_RT']

        matmul_in_config = dict({
            'GLOBAL_M_VALUE': self._tgt_seq_len,
            'GLOBAL_K_VALUE': self._embed_dim,
            'GLOBAL_N_VALUE': self._src_seq_len,
        }, **matmul_in_params)
        softmax_config = dict({
            'GLOBAL_H_VALUE': self._tgt_seq_len,
            'GLOBAL_W_VALUE': self._src_seq_len,
        }, **softmax_params)
        matmul_out_config = dict({
            'GLOBAL_M_VALUE': self._tgt_seq_len,
            'GLOBAL_K_VALUE': self._src_seq_len,
            'GLOBAL_N_VALUE': self._embed_dim,
        }, **matmul_out_params)
        return matmul_in_config, softmax_config, matmul_out_config

    def build(self, params: Dict, sample_inputs: List, jit: bool = True):
        matmul_in_config, softmax_config, matmul_out_config = self._read_params(params)
        matmul_in_forward_kernel = self._matmul_in_possible_implementations[matmul_in_config['_name']]
        self._matmul_in_forward_function = matmul_in_forward_kernel.compile(
            matmul_in_config, {'C': self._mask}, jit
        ).forward
        softmax_forward_kernel = self._softmax_possible_implementations[softmax_config['_name']]
        self._softmax_forward_function = softmax_forward_kernel.compile(
            softmax_config, {'C_in': self._mask, 'C_mask': self._mask, 'C_out': self._mask}, jit
        ).forward
        compressed_mask = softmax_forward_kernel.get_input('C_mask').sparse()['val']
        self._mask_cuda = torch.from_numpy(compressed_mask).cuda()
        matmul_out_forward_kernel = self._matmul_out_possible_implementations[matmul_out_config['_name']]
        self._matmul_out_forward_function = matmul_out_forward_kernel.compile(
            matmul_out_config, {'A': self._mask}, jit
        ).forward
        self.ready = True

    def _reshape_input(self, x: torch.Tensor):
        # x = x.reshape((self._batch_size, -1, self._num_heads, self._embed_dim // self._num_heads))
        # x = x.swapaxes(1, 2).contiguous()
        # x = x.reshape((self._batch_size * self._num_heads, -1, self._embed_dim // self._num_heads))
        # return x
        return x.flatten(0, 1)

    def _reshape_output(self, x: torch.Tensor):
        # x = x.reshape((self._batch_size, self._num_heads, -1, self._embed_dim // self._num_heads))
        # x = x.swapaxes(1, 2).contiguous()
        # x = x.reshape((self._batch_size, -1, self._embed_dim))
        # return x
        return x.reshape((self._batch_size, self._num_heads, self._tgt_seq_len, self._embed_dim))

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        Q = self._reshape_input(Q)
        K = self._reshape_input(K)
        V = self._reshape_input(V)
        if self.ready:
            QK = self._matmul_in_forward_function(Q, K)
            QK_softmax = self._softmax_forward_function(QK, self._mask_cuda)
            result = self._matmul_out_forward_function(QK_softmax, V)
        else:
            warnings.warn('the sparse module is not compiled, using dense functions to forward')
            QK = torch.bmm(Q, K.swapaxes(1, 2))
            # QK_softmax = torch.softmax(QK, dim=-1)
            QK_softmax = sparse_softmax_reference(QK, self._mask_cuda)
            result = torch.bmm(QK_softmax, V)
        return self._reshape_output(result)

    def _forward_kernel_test(
        self, params: Dict, sample_inputs: List[torch.Tensor],
        num_warmups: int = 10, num_iters: int = 10
    ):
        Q, K, V = sample_inputs
        assert Q.shape == (self._batch_size, self._num_heads, self._tgt_seq_len, self._embed_dim)
        assert K.shape == (self._batch_size, self._num_heads, self._src_seq_len, self._embed_dim)
        assert V.shape == (self._batch_size, self._num_heads, self._src_seq_len, self._embed_dim)

        matmul_in_config, softmax_config, matmul_out_config = self._read_params(params)

        Q_numpy = self._reshape_input(Q).cpu().detach().unsqueeze(0).numpy().astype('float32')
        K_numpy = self._reshape_input(K).cpu().detach().unsqueeze(0).numpy().astype('float32')
        V_numpy = self._reshape_input(V).cpu().detach().unsqueeze(0).numpy().astype('float32')

        matmul_in_kernel = self._matmul_in_possible_implementations[matmul_in_config['_name']]
        matmul_in_kernel.set_input('A', Q_numpy)
        matmul_in_kernel.set_input('B', K_numpy)
        QK = matmul_in_kernel.calc_target_outputs()['C']
        latency = matmul_in_kernel.test(
            config=matmul_in_config,
            mask={'C': self._mask},
            target_outputs={'C': QK},
            num_warmups=num_warmups,
            num_iters=num_iters,
        )

        softmax_kernel = self._softmax_possible_implementations[softmax_config['_name']]
        softmax_kernel.set_input('C_in', QK)
        softmax_kernel.set_input('C_mask', self._mask.astype('int32'))
        QK_softmax = softmax_kernel.calc_target_outputs()['C_out']
        latency += softmax_kernel.test(
            config=softmax_config,
            mask={'C_in': self._mask, 'C_mask': self._mask, 'C_out': self._mask},
            target_outputs={'C': QK},
            num_warmups=num_warmups,
            num_iters=num_iters,
        )

        matmul_out_kernel = self._matmul_out_possible_implementations[matmul_out_config['_name']]
        matmul_out_kernel.set_input('A', QK_softmax)
        matmul_out_kernel.set_input('B', V_numpy)
        latency += matmul_out_kernel.test(
            config=matmul_out_config,
            mask={'A': self._mask},
            num_warmups=num_warmups,
            num_iters=num_iters,
        )
        return latency

    def set_search_space(self, search_space: TunableItemCfg = None):
        '''
        MAT_IN   (BM) BK [BN] TM TK TN
        SOFTMAX  (BH) [BW] RT
        MAT_OUT  (BM) [BK] BN TM TK TN
        '''
        if search_space is None:
            search_space = TunableItemCfg(
                'choice',
                _is_nested=True,
                _value={
                    'sparta_sparta_sparta': {
                        'BH': [8, 16, 32, 64],
                        'BW': [8, 16, 32, 64],
                        'MATMUL_IN_BK': [32, 64],
                        'MATMUL_IN_TM': [4],
                        'MATMUL_IN_TK': [4],
                        'MATMUL_IN_TN': [4],
                        'SOFTMAX_RT': [4],
                        'MATMUL_OUT_BN': [32, 64],
                        'MATMUL_OUT_TM': [4],
                        'MATMUL_OUT_TK': [4],
                        'MATMUL_OUT_TN': [4],
                    },
                    'openai_sparta_sparta': {
                        'SOFTMAX_RT': [4],
                        'MATMUL_OUT_BN': [32, 64],
                        'MATMUL_OUT_TM': [4],
                        'MATMUL_OUT_TK': [4],
                        'MATMUL_OUT_TN': [4],
                    },
                    'sparta_sparta_openai': {
                        'MATMUL_IN_BK': [32, 64],
                        'MATMUL_IN_TM': [4],
                        'MATMUL_IN_TK': [4],
                        'MATMUL_IN_TN': [4],
                        'SOFTMAX_RT': [4],
                    },
                }
            )
        self._search_space = search_space

    def tester(
        self, params: Dict, sample_inputs: List, jit: bool = False, weight_bk: float=0.0,
        num_warmups: int = 10, num_iters: int = 10
    ) -> float:
        if jit:
            self.build(params, sample_inputs, jit)
            # how to get the latency of the compiled kernel?
            latency = test_latency(self.forward, sample_inputs, None, num_warmups, num_iters)
            if weight_bk > 0:
                # TODO add backward time
                raise NotImplementedError
        else:
            latency = self._forward_kernel_test(params, sample_inputs, num_warmups, num_iters)
            if weight_bk > 0:
                # TODO add backward time
                raise NotImplementedError
        return latency
