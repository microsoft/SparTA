# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Dict, Tuple, Optional

import torch
import numpy as np

from sparta.specializer.kernels import SparTASparseMatMulKernel, OpenAISparseMatMulKernel, SparTASparseSoftmaxKernel, SparTASparseSoftmaxBackwardKernel
from sparta.specializer.funtional import SparseCtxBase, KernelPlaceholder


class SparseMultiHeadAttentionCtx(SparseCtxBase):

    def __init__(self):
        super().__init__()

        self._T = np.float32(1.)
        self._mask: torch.Tensor = None
        self._tesa_shapes: Dict[str, Tuple[str, str]] = {}

        for kernel_name, s_type, trans_A, trans_B, sparse_tensor, tesa_shape in zip(
            ['forward:qk', 'forward:out', 'backward:v', 'backward:sm', 'backward:q', 'backward:k'],
            ['dds', 'sdd', 'sdd', 'dds', 'sdd', 'sdd'],
            [False, False, True, False, False, True],
            [True, False, False, True, False, False],
            ['C', 'A', 'A', 'C', 'A', 'A'],
            [('M', 'N'), ('M', 'K'), ('M', 'K'), ('M', 'N'), ('M', 'K'), ('M', 'K')]
        ):
            self._kernels[kernel_name] = KernelPlaceholder(
                name=kernel_name,
                impls={
                    'sparta': SparTASparseMatMulKernel,
                    'openai': OpenAISparseMatMulKernel,
                },
                args={
                    'biased': False,
                    'bcsr_main': 'H',
                    'compressed': True,
                    'transpose_A': trans_A,
                    'transpose_B': trans_B,
                    'sparse_type': s_type,
                },
                mask_map={'qk': sparse_tensor},
            )
            self._tesa_shapes[kernel_name] = tesa_shape

        for kernel_name, kernel_class, first_tensor in zip(
            ['forward:sm', 'backward:qk'],
            [SparTASparseSoftmaxKernel, SparTASparseSoftmaxBackwardKernel],
            ['x', 'grad_y'],
        ):
            self._kernels[kernel_name] = KernelPlaceholder(
                name=kernel_name,
                impls={'sparta': kernel_class},
                args={'compressed': True},
                mask_map={'qk': first_tensor},
            )
            self._tesa_shapes[kernel_name] = ('H', 'W')

    def set_shape(self, batch_size: int, src_seq_len: int, tgt_seq_len: int, embed_dim: int):
        self._kernels['forward:qk'].set_shape(batch_size, tgt_seq_len, embed_dim, src_seq_len)
        self._kernels['forward:sm'].set_shape(batch_size, tgt_seq_len, src_seq_len)
        self._kernels['forward:out'].set_shape(batch_size, tgt_seq_len, src_seq_len, embed_dim)
        self._kernels['backward:v'].set_shape(batch_size, src_seq_len, tgt_seq_len, embed_dim)
        self._kernels['backward:sm'].set_shape(batch_size, tgt_seq_len, embed_dim, src_seq_len)
        self._kernels['backward:qk'].set_shape(batch_size, tgt_seq_len, src_seq_len)
        self._kernels['backward:q'].set_shape(batch_size, tgt_seq_len, src_seq_len, embed_dim)
        self._kernels['backward:k'].set_shape(batch_size, src_seq_len, tgt_seq_len, embed_dim)
        self._T = np.float32(1 / np.sqrt(embed_dim))

    def get_conditions(self, impls: Dict[str, str]):
        conditions = [[], []]
        fixed_dims = [None, None]
        for kernel_name, impl in impls.items():
            for k, dim in enumerate(self._tesa_shapes[kernel_name]):
                if impl == 'sparta':
                    v = f'{kernel_name};BLOCK_SIZE_{dim}_VALUE'
                else:  # impl == 'openai'
                    v = {'M': 32, 'K': 64, 'N': 32}[dim]
                    if fixed_dims[k] is None:
                        fixed_dims[k] = v
                    elif fixed_dims[k] != v:
                        return None
                conditions[k].append(v)
        return conditions

    def build(self, config: Dict[str, Any], mask: Dict[str, torch.Tensor]):
        super().build(config, mask)
        self._mask = self.get_converter('forward:qk', 'qk').convert(mask['qk']).to(torch.int32)

    def _split_graph(
        self, kernels: List[str], sample_inputs: Dict[str, torch.Tensor],
        sample_grad: Optional[torch.Tensor] = None
    ):
        funcs, inputs = [], []
        q, k, v = sample_inputs['q'], sample_inputs['k'], sample_inputs['v']
        qk = self.forward_qk(sample_inputs['q'], sample_inputs['k'])
        sm = self.forward_sm(qk)
        grad_sm = self.backward_sm(sample_grad, sample_inputs['v'])
        grad_qk = self.backward_qk(grad_sm, sm)
        if 'forward:qk' in kernels:
            funcs.append(self.forward_qk)
            inputs.append([q, k])
        if 'forward:sm' in kernels:
            funcs.append(self.forward_sm)
            inputs.append([qk])
        if 'forward:out' in kernels:
            funcs.append(self.forward_out)
            inputs.append([sm, v])
        if 'backward:v' in kernels:
            funcs.append(self.backward_v)
            inputs.append([sample_grad, sm])
        if 'backward:sm' in kernels:
            funcs.append(self.backward_sm)
            inputs.append([sample_grad, v])
        if 'backward:qk' in kernels:
            funcs.append(self.backward_sm)
            inputs.append([grad_sm, sm])
        if 'backward:q' in kernels:
            funcs.append(self.backward_q)
            inputs.append([grad_qk, k])
        if 'backward:k' in kernels:
            funcs.append(self.backward_k)
            inputs.append([grad_qk, q])
        return funcs, inputs

    def forward_qk(self, q: torch.Tensor, k: torch.Tensor):
        return self._kernels['forward:qk'].active_kernel()(q, k)

    def forward_sm(self, qk: torch.Tensor):
        return self._kernels['forward:sm'].active_kernel()(qk, self._mask, self._T)

    def forward_out(self, sm: torch.Tensor, v: torch.Tensor):
        return self._kernels['forward:out'].active_kernel()(sm, v)

    def backward_v(self, grad_out: torch.Tensor, sm: torch.Tensor):
        return self._kernels['backward:v'].active_kernel()(sm, grad_out)

    def backward_sm(self, grad_out: torch.Tensor, v: torch.Tensor):
        return self._kernels['backward:sm'].active_kernel()(grad_out, v)

    def backward_qk(self, grad_sm: torch.Tensor, sm: torch.Tensor):
        return self._kernels['backward:qk'].active_kernel()(grad_sm, sm, self._mask, self._T)

    def backward_q(self, grad_qk: torch.Tensor, k: torch.Tensor):
        return self._kernels['backward:q'].active_kernel()(grad_qk, k)

    def backward_k(self, grad_qk: torch.Tensor, q: torch.Tensor):
        return self._kernels['backward:k'].active_kernel()(grad_qk, q)


class SparseMultiHeadAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, sparta_ctx: SparseMultiHeadAttentionCtx,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        ctx.sparta_ctx = sparta_ctx
        qk = sparta_ctx.forward_qk(q.detach(), k.detach())
        sm = sparta_ctx.forward_sm(qk)
        ctx.save_for_backward(q, k, v, sm)
        return sparta_ctx.forward_out(sm, v.detach())

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor):
        q, k, v, sm = ctx.saved_tensors
        grad_q = grad_k = grad_v = None
        converter = ctx.sparta_ctx.get_converter('forward:qk', 'qk')
        if ctx.needs_input_grad[3]:
            grad_v = ctx.sparta_ctx.backward_v(grad_out.detach(), converter.swapaxes(sm))
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            grad_sm = ctx.sparta_ctx.backward_sm(grad_out.detach(), v.detach())
            grad_qk = ctx.sparta_ctx.backward_qk(grad_sm, sm)
            if ctx.needs_input_grad[1]:
                grad_q = ctx.sparta_ctx.backward_q(grad_qk, k.detach())
            if ctx.needs_input_grad[2]:
                grad_k = ctx.sparta_ctx.backward_k(converter.swapaxes(grad_qk), q.detach())

        return None, grad_q, grad_k, grad_v
