# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional
import warnings

import torch
import numpy as np

from sparta.specializer.operators import OperatorBase, SparseBatchMatMul, SparseSoftmax


class SparseAttention(OperatorBase):

    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.mask = mask
        self._Nt, self._Ns = mask.shape
        self._matmul_qk = SparseBatchMatMul(C_mask=mask, transpose_A=False, transpose_B=True, compressed=True)
        self._softmax = SparseSoftmax(mask=mask, compressed=True)
        self._matmul_out = SparseBatchMatMul(A_mask=mask, transpose_A=False, transpose_B=False, compressed=True)
        self.ready: bool = False

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        if not self.ready:
            self._softmax.set_temperature(np.sqrt(query.shape[-1]))
        qk = self._matmul_qk.forward(query, key)
        sm = self._softmax.forward(qk)
        out = self._matmul_out.forward(sm, value)
        return out

    def build(self, config: Dict[str, Any], sample_inputs: List[torch.Tensor]):
        query, key, value = sample_inputs

        qB = np.prod(query.shape[:-2])
        qN, qE = query.shape[-2:]
        kB = np.prod(key.shape[:-2])
        kN, kE = key.shape[-2:]
        vB = np.prod(value.shape[:-2])
        vN, vE = value.shape[-2:]
        assert qB == kB == vB, f'query, key and value should have the same batch size'
        assert self._Nt == qN, f'expect query shape (?, {self._Nt}, ?), got {query.shape}'
        assert self._Ns == kN, f'expect key shape (?, {self._Ns}, ?), got {key.shape}'
        assert self._Ns == vN, f'expect value shape (?, {self._Ns}, ?), got {value.shape}'
        assert qE == kE == vE, f'query, key and value should have the same embed dim'

        self._softmax.set_temperature(np.sqrt(qE))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            qk = self._matmul_qk.forward(query, key)
            sm = self._softmax.forward(qk)

        self._matmul_qk.build(
            config={
                k.split('/')[1]: v
                for k, v in config.items()
                if k.startswith('qk')
            },
            sample_inputs=[query, key],
        )
        self._softmax.build(
            config={
                k.split('/')[1]: v
                for k, v in config.items()
                if k.startswith('sm')
            },
            sample_inputs=[qk],
        )
        self._matmul_out.build(
            config={
                k.split('/')[1]: v
                for k, v in config.items()
                if k.startswith('out')
            },
            sample_inputs=[sm, value],
        )

        self.ready = True

    def set_sample_inputs(
        self,
        sample_inputs: List[torch.Tensor],
        sample_grads: Optional[List[torch.Tensor]] = None,
    ):
        query, key, value = sample_inputs

        if sample_grads is not None:
            query.requires_grad = True
            key.requires_grad = True
            value.requires_grad = True

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            qk = self._matmul_qk.forward(query, key)
            sample_grad_qk = None
            sm = self._softmax.forward(qk)
            sample_grad_sm = None

        if sample_grads is not None:
            qk.retain_grad()
            sm.retain_grad()
            grad_out, = sample_grads

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out = self._matmul_out.forward(sm, value)
                out.backward(grad_out)

            sample_grad_sm = [sm.grad]
            sample_grad_qk = [qk.grad]
            query = query.detach()
            key = key.detach()
            value = value.detach()

        self._matmul_qk.set_sample_inputs([query, key], sample_grad_qk)
        self._softmax.set_sample_inputs([qk], sample_grad_sm)
        self._matmul_out.set_sample_inputs([sm, value], sample_grads)

    def _combine_dict(
        self,
        qk_dict: Dict[str, Any],
        sm_dict: Dict[str, Any],
        out_dict: Dict[str, Any],
        backward: bool = False,
    ):
        return dict(
            **{f'qk/{k}': v for k, v in qk_dict.items() if k.startswith('forward') or backward},
            **{f'sm/{k}': v for k, v in sm_dict.items() if k.startswith('forward') or backward},
            **{f'out/{k}': v for k, v in out_dict.items() if k.startswith('forward') or backward},
        )

    def get_search_space(self, backward: bool = False):
        return self._combine_dict(
            qk_dict=self._matmul_qk.get_search_space(backward=True),
            sm_dict=self._softmax.get_search_space(backward=True),
            out_dict=self._matmul_out.get_search_space(backward=True),
            backward=backward,
        )

    def get_connections(self, backward: bool = False):
        return [
            self._combine_dict(
                qk_dict=qk_params,
                sm_dict=sm_params,
                out_dict=out_params,
                backward=backward,
            )
            for qk_params, sm_params, out_params in zip(
                self._matmul_qk.get_connections(backward=True),
                self._softmax.get_connections(backward=True),
                self._matmul_out.get_connections(backward=True),
            )
        ]

    def get_kernel_placeholders(self, backward: bool = False):
        return self._combine_dict(
            qk_dict=self._matmul_qk.get_kernel_placeholders(backward=True),
            sm_dict=self._softmax.get_kernel_placeholders(backward=True),
            out_dict=self._matmul_out.get_kernel_placeholders(backward=True),
            backward=backward,
        )
