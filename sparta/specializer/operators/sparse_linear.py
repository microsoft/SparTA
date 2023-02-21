# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional

import torch

from sparta.specializer.operators import OperatorBase
from sparta.specializer.functional import SparseBatchMatMul


class SparseLinear(OperatorBase):
    r"""The sparse linear operator: :math:`y = xA^T + b`

    Args:
        raw_module (torch.nn.Linear): The corresponding dense linear operator.
        input_mask (Optional[torch.Tensor]): The mask of input tensor.
            If `input_mask` is set, the other two masks should be `None`
            and the internal MatMul kernel will choose SD=>D mode.
        weight_mask (Optional[torch.Tensor]): The mask of weight tensor.
            If `weight_mask` is set, the other two masks should be `None`
            and the internal MatMul kernel will choose DS=>D mode.
        output_mask (Optional[torch.Tensor]): The mask of output tensor.
            If `output_mask` is set, the other two masks should be `None`
            and the internal MatMul kernel will choose DD=>S mode.

    Shape:
        - Input: :math:`(B, H_{in})` where :math:`B = \text{batch_size}`
            and :math:`H_{in} = \text{in_features}`.
        - Output: :math:`(B, H_{out})` where :math:`H_{out} = \text{out_features}`.

    Attributes:
        weight: The learnable weights of the module of shape :math:`(\text{out_features}, \text{in_features})`.
            If `weight_mask` is set, the weight will be compressed to BCSR format.
        bias: The learnable bias of the module of shape :math:`(\text{out_features})`.
            It is a copy of the bias tensor in the raw module.

    Examples:

        .. code-block:: python

            batch_size, in_features, out_features = 1024, 1024, 1024

            # Create a dense linear operator
            dense_linear = torch.nn.Linear(in_features, out_features, device='cuda')

            # Create a weight mask
            mask = sparta.testing.block_mask((out_features, in_features), sparsity=0.99)

            # Create a sparse linear operator using the dense operator and the weight mask
            sparse_linear = sparta.nn.SparseLinear(dense_linear, weight_mask=mask)

            # Tune the sparse linear operator
            sparta.nn.tune(sparse_linear, sample_inputs=[
                torch.rand((batch_size, in_features), device='cuda'),
            ])

    """

    __base_class__ = torch.nn.Linear
    __sparse_func__ = SparseBatchMatMul

    def __init__(
        self,
        raw_module: torch.nn.Linear,
        input_mask: Optional[torch.Tensor] = None,
        weight_mask: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__(raw_module)

        M = None
        N, K = raw_module.weight.shape
        biased = raw_module.bias is not None
        self._raw_weight = torch.clone(raw_module.weight)
        self.weight = None
        self.bias = raw_module.bias

        if input_mask is not None:
            self._sparse_ctx = SparseBatchMatMulCtx('sdd', False, True, biased, False)
            M = input_mask.shape[0]
        elif weight_mask is not None:
            self._sparse_ctx = SparseBatchMatMulCtx('dsd', False, True, biased, True)
        elif output_mask is not None:
            self._sparse_ctx = SparseBatchMatMulCtx('dds', False, True, biased, False)
            M = output_mask.shape[0]
        else:
            raise ValueError(f'expected a sparse mask on input / weight / output')

        self._shape = {'batch_size': 1, 'M': M, 'K': K, 'N': N}
        self.update_mask(input_mask, weight_mask, output_mask)

    def update_mask(
        self,
        input_mask: Optional[torch.Tensor] = None,
        weight_mask: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None,
    ):
        if sum(map(lambda x: x is not None, [input_mask, weight_mask, output_mask])) > 1:
            raise ValueError(f'linear operators with multiple sparse masks are not supported')

        M, K, N = self._shape['M'], self._shape['K'], self._shape['N']

        def check_mask_shape(mask: torch.Tensor, name: str, H: Optional[int], W: Optional[int]):
            if H is None:
                check_H = lambda x: True
                H = '?'
            else:
                check_H = lambda x: x == H
            if W is None:
                check_W = lambda x: True
                W = '?'
            else:
                check_W = lambda x: x == W
            err_msg = f'expected {name} mask shape ({H}, {W}), got {mask.shape}'
            assert check_H(mask.shape[0]) and check_W(mask.shape[1]), err_msg

        if input_mask is not None:
            check_mask_shape(input_mask, 'input', M, K)
            self._set_mask({'A': input_mask})
        elif weight_mask is not None:
            check_mask_shape(weight_mask, 'weight', N, K)
            self._set_mask({'B': weight_mask})
        elif output_mask is not None:
            check_mask_shape(output_mask, 'output', M, N)
            self._set_mask({'C': output_mask})
        else:
            raise ValueError(f'expected a sparse mask on input / weight / output')

        if self.ready:
            self._update_parameters()

    def _read_sample_inputs(self, A: torch.Tensor):
        M, K = self._shape['M'], self._shape['K']
        if M is None:
            assert K == A.shape[1], f'expect input shape (?, {K}), got {A.shape}'
            self._shape['M'] = A.shape[0]
        else:
            assert (M, K) == A.shape, f'expect input shape ({M}, {K}), got {A.shape}'

    def _update_parameters(self):
        weight = self._raw_weight
        weight_indexes = self.get_sparse_indexes('B')
        if weight_indexes is not None:
            weight = weight_indexes.convert(weight.detach())
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def _compile(self, config: Optional[Dict[str, Dict[str, Any]]] = None):
        super()._compile(config)
        self._update_parameters()

    def _sparse_forward(self, input_tensor: torch.Tensor):
        inputs = [self._sparse_ctx, input_tensor, self.weight]
        if self.bias is not None:
            inputs.append(self.bias)
        return self.__sparse_func__.apply(*inputs).squeeze(0)

    def set_sample_inputs(
        self,
        sample_inputs: List[torch.Tensor],
        sample_grads: Optional[List[torch.Tensor]] = None,
    ):
        self._read_sample_inputs(*sample_inputs)
        self._sparse_ctx.set_shape(**self._shape)
        if self.bias is None:
            sample_inputs = [sample_inputs[0], self._raw_weight]
        else:
            sample_inputs = [sample_inputs[0], self._raw_weight, self.bias]
        sample_inputs = [x.unsqueeze(0) for x in sample_inputs]
        if sample_grads is not None:
            sample_grads = [x.unsqueeze(0) for x in sample_grads]
        self._sparse_ctx.set_sample_inputs(sample_inputs, sample_grads)
