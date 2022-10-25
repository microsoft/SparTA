
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch
import numpy as np

from sparta.specializer.operators import OperatorBase
from sparta.specializer.funtional import SparseMultiHeadAttentionCtx, SparseMultiHeadAttention


class SparseAttention(OperatorBase):

    __sparse_func__ = SparseMultiHeadAttention

    def __init__(self, mask: torch.Tensor):
        super().__init__()
        Nt, Ns = mask.shape
        self._sparse_ctx = SparseMultiHeadAttentionCtx()
        self._mask = {'qk': mask}
        self._shape = {'src_seq_len': Ns, 'tgt_seq_len': Nt}

    def _read_sample_inputs(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        Ns, Nt = self._shape['src_seq_len'], self._shape['tgt_seq_len']
        qB = int(np.prod(query.shape[:-2]))
        qN, qE = query.shape[-2:]
        kB = int(np.prod(key.shape[:-2]))
        kN, kE = key.shape[-2:]
        vB = int(np.prod(value.shape[:-2]))
        vN, vE = value.shape[-2:]
        assert qB == kB == vB, f'query, key and value should have the same batch size'
        assert Nt == qN, f'expect query shape (?, {Nt}, ?), got {query.shape}'
        assert Ns == kN, f'expect key shape (?, {Ns}, ?), got {key.shape}'
        assert Ns == vN, f'expect value shape (?, {Ns}, ?), got {value.shape}'
        assert qE == kE == vE, f'query, key and value should have the same embed dim'
        self._shape['batch_size'] = qB
        self._shape['embed_dim'] = qE

    def _sparse_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        return self.__sparse_func__.apply(self._sparse_ctx, query, key, value)

    def _construct_inputs(self, raw_inputs: List[torch.Tensor]):
        return {'q': raw_inputs[0], 'k': raw_inputs[1], 'v': raw_inputs[2]}
