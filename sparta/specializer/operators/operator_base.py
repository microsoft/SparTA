# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import sys
import warnings
import subprocess
from typing import Type, Tuple, List, Dict, Union

import torch
import numpy as np

from sparta.specializer import kernels
from sparta.common.tuning import TunableItemCfg, Tunable
from sparta.common.utils import get_uname


class OperatorBase(torch.nn.Module):
    '''Base class of sparse operators.

    Examples:

        .. code-block:: python

            # Create a dense softmax layer
            dense_softmax = torch.nn.Softmax

            # Create a mask
            mask = torch.rand((2048, 1024)) > 0.99

            # Create a sparse softmax layer using the dense layer and the mask
            sparse_softmax = sparta.nn.SparseSoftmax(dense_softmax, mask=mask)

            # Tune the sparse softmax layer
            sparta.tune(sparse_softmax, sample_inputs=[torch.rand((2048, 1024))])

    Args:
        raw_module (torch.nn.Module): The corresponding dense operator.
    '''
    __base_class__: Type[torch.nn.Module] = None

    def __init__(self, raw_module: torch.nn.Module):
        if type(raw_module) is not self.__base_class__:
            raise ValueError(f'expected a {self.__base_class__} module')
        super().__init__()
        self._raw_module = raw_module
        self._forward_function = None
        self._mask = None
        self.ready = False
        self._possible_implementations = {}
        self._search_space = None

    def build(self, params: Dict, sample_inputs: List, jit: bool = True):
        '''Build the sparse kernel using the specified implementation and configs.

        Args:
            params (Dict): building parameters. It should be a valid sample of search space
                params['_name'] should be a valid kernel name in `self._possible_implementations`
                other key-value pairs in params are the parameters for `self._possible_implementations[params['_name']]`
            sample_inputs (List): sample inputs for shape inference
            jit (bool): Determine whether to build the kernel using JIT mode.
        '''
        if sample_inputs:
            shape, inputs = self._read_sample_inputs(*sample_inputs)
        forward_kernel = self._possible_implementations[params['_name']]
        self._forward_function = forward_kernel.compile(params, self._mask, jit).forward
        self._load_compile_kernel(forward_kernel)
        self.ready = True

    def forward(self, *args):
        '''Forward function. Calls the corresponding dense operator if not built.'''
        if self.ready:
            return self._sparse_forward(*args)
        else:
            warnings.warn('the sparse module is not compiled, using the dense module to forward')
            return self._raw_module.forward(*args)

    @abc.abstractmethod
    def _sparse_forward(self, *args):
        '''Calls the sparse forward kernel.'''

    @abc.abstractmethod
    def _load_compile_kernel(self, forward_kernel: kernels.KernelBase):
        '''Set PyTorch module parameters according to the dense operator.

        Args:
            forward_kernel (kernels.KernelBase): The forward kernel object
                which provides the sparsify function.
        '''

    @abc.abstractmethod
    def _read_sample_inputs(self, *args) -> Tuple[dict, dict]:
        '''Read shape config and convert sample inputs to test inputs.'''

    def set_search_space(self, search_space: TunableItemCfg = None):
        '''Input a custom search space to override the default one before tuning.

        Examples:

            .. code-block:: python

                # Create a dense linear layer
                dense_linear = torch.nn.Linear(1024, 2048)

                # Create a mask
                weight_mask = torch.rand((2048, 1024)) > 0.99

                # Create a sparse linear layer using the dense layer and the mask
                sparse_linear = sparta.nn.SparseLinear(dense_linear, weight_mask=weight_mask)

                # Set custom search space
                search_space_cfg = TunableItemCfg('choice', {
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
                sparse_linear.set_search_space(search_space_cfg)

                # Tune the sparse linear layer
                sparta.tune(sparse_linear, sample_inputs=[torch.rand((512, 1024))])

        Args:
            search_space (dict): Key is the tuning algorithm, value is a dictionary whose keys are
                tunable parameters and values are lists of possible values.
        '''
        if search_space is None:
            search_space = TunableItemCfg(
                'choice',
                _is_nested=True,
                _value={k: v.get_search_space() for k, v in self._possible_implementations.items()}
            )
        self._search_space = search_space

    def get_search_space(self) -> TunableItemCfg:
        '''Get the search space of the sparse operator.

        Returns:
            TunableItemCfg: the search space of the sparse operator.
        '''
        if self._search_space is None:
            self.set_search_space()
        return self._search_space

    def tester(self, params: Dict,  sample_inputs: List, jit: bool = False, weight_bk: float=0.) -> float:
        '''Tester function for tuning. It will build the sparse kernel and run the forward function (or backward also), and return the measured time.

        Args:
            params (Dict): building parameters. It should be a valid sample of search space
            sample_inputs (List): sample inputs for shape inference
            jit (bool): Determine whether to test the kernel using JIT mode.
            weight_bk (float): The weight of the backward time in the total time. If set to 0, the backward time is not counted.

        Returns:
            float: The performance (running latency) of the kernel.
        '''
        if jit:
            self.build(params, sample_inputs, jit)
            # how to get the latency of the compiled kernel?
            raise NotImplementedError
        else:
            shape, inputs = self._read_sample_inputs(*sample_inputs)
            implement, cfg = params['_name'], params
            kernel = self._possible_implementations[implement]
            latency = kernel.test(dict(shape, **cfg), mask=self._mask, inputs=inputs)
            if weight_bk > 0:
                # TODO add backward time
                pass
            return latency

