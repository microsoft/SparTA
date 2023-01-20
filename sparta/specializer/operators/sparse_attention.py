# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional
import warnings

import torch
import numpy as np

from sparta.specializer.operators import OperatorBase, SparseBatchMatMul, SparseSoftmax
from sparta.specializer.kernels import KernelBase, PortConfig


class SparseAttention(OperatorBase):
    r"""The sparse attention operator.

    .. math::
        \text{Attention}(Q, K, V) = \text{Softmax}(Q K) V

    Args:
        mask (torch.Tensor): The mask tensor of shape :math:`(N_{target}, N_{sourse})`,
            where :math:`N_{target}` is the target sequence length
            and :math:`N_{sourse}` is the sourse sequence length.

    Shape:
        - Input1: :math:`(B * H, N_{target}, E)` where :math:`B` is the batch size,
            :math:`H` is the number of heads and :math:`E` is the embed dimension.
        - Input2: :math:`(B * H, N_{sourse}, E)`.
        - Input3: :math:`(B * H, N_{sourse}, E)`, same shape as the second input.
        - Output: :math:`(B * H, N_{target}, E)`, same shape as the first input.

    Examples:

        .. code-block:: python
    
            B, H, Ns, Nt, E = 4, 4, 1024, 1024, 1024

            # Create a mask
            mask = sparta.testing.block_mask((Nt, Ns), sparsity=0.99)

            # Create a sparse attention operator using the mask
            sparse_attention = sparta.nn.SparseAttention(mask=mask)

            # Tune the sparse attention operator
            sparta.nn.tune(sparse_attention, sample_inputs=[
                torch.rand((B * H, Nt, E), device='cuda'),
                torch.rand((B * H, Ns, E), device='cuda'),
                torch.rand((B * H, Ns, E), device='cuda'),
            ])

    """

    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.mask = mask
        self._Nt, self._Ns = mask.shape
        self._matmul_qk = SparseBatchMatMul(
            C_mask=mask,
            transpose_A=False,
            transpose_B=True,
            compressed=True,
        )
        self._softmax = SparseSoftmax(
            mask=mask,
            compressed=True,
        )
        self._matmul_out = SparseBatchMatMul(
            A_mask=mask,
            transpose_A=False,
            transpose_B=False,
            compressed=True,
        )
        self._sparse_port: PortConfig = None
        self.ready: bool = False

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        qk = self._matmul_qk.forward(query, key)
        sm = self._softmax.forward(qk)
        out = self._matmul_out.forward(sm, value)
        return out

    def update_mask(self, mask: torch.Tensor):
        if self.ready:
            self._sparse_port.set_mask(mask)
            self._matmul_qk._sparse_ctx.update_func()
            self._softmax._sparse_ctx.update_func()
            self._matmul_out._sparse_ctx.update_func()
        else:
            op_list: List[OperatorBase] = [self._matmul_qk, self._softmax, self._matmul_out]
            for op in op_list:
                for ports in op._sparse_ctx.sparse_ports.values():
                    for port in ports:
                        port.set_mask(mask)

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

        op_dict: Dict[str, OperatorBase] = {
            'qk': self._matmul_qk,
            'sm': self._softmax,
            'out': self._matmul_out,
        }
        inputs_dict: Dict[str, List[torch.Tensor]] = {
            'qk': [query, key],
            'sm': [qk],
            'out': [sm, value],
        }
        sparse_port_map: Dict[str, str] = {
            'qk': 'C',
            'sm': 'y',
            'out': 'A',
        }

        kernels: List[KernelBase] = []
        port_names: List[str] = []
        for op_name, op in op_dict.items():
            for k, v in op._sparse_ctx.get_kernel_placeholders().items():
                kernel = v.possible_kernels[config[f'{op_name}/{k}']['_impl']]
                port_names.append(v.port_map[sparse_port_map[op_name]])
                kernels.append(kernel)
        self._sparse_port = kernels[0].ports[port_names[0]]
        for kernel, port_name in zip(kernels[1:], port_names[1:]):
            self._sparse_port.connect(kernel, port_name)

        for op_name, op in op_dict.items():
            op.build(
                config={
                    k.split('/')[1]: v
                    for k, v in config.items()
                    if k.startswith(op_name)
                },
                sample_inputs=inputs_dict[op_name],
            )

        self.ready = True

    def set_sample_inputs(
        self,
        sample_inputs: List[torch.Tensor],
        sample_grads: Optional[List[torch.Tensor]] = None,
    ):
        query, key, value = sample_inputs
        self._softmax.set_temperature(np.sqrt(query.shape[-1]))

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
