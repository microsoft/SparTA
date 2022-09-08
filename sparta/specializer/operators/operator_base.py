# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from sys import implementation
import warnings
import subprocess
from typing import Type, Tuple, List, Dict

import torch
import numpy as np

from sparta.specializer import kernels, tuners
from sparta.common.tuning import TunableItemCfg, Tunable


class OperatorBase(torch.nn.Module):
    '''Base class of sparse operators.

    Args:
        raw_module (torch.nn.Module): The corresponding dense operator.
        base_class (Type[torch.nn.Module]): Class of the dense operator.
    '''

    def __init__(self, raw_module: torch.nn.Module, base_class: Type[torch.nn.Module]):
        if type(raw_module) is not base_class:
            raise ValueError(f'expected a {base_class} module')
        super().__init__()
        self._raw_module = raw_module
        self._forward_kernel = None
        self._forward_function = None
        self._tuner = None
        self._mask = None
        self.ready = False
        self._possible_implementations = {}

    def build(self, params: Dict, sample_inputs: List = None, jit: bool = True):
        '''Build the sparse kernel using the specified implementation and configs.

        Args:
            params (str): building parameters. It should be a valid sample of search space
            jit (bool): Determine whether to build the kernel using JIT mode.
        '''
        if sample_inputs:
            shape, inputs = self._read_sample_inputs(*sample_inputs)
        forward_kernel = self._possible_implementations[params['implement']]
        self._forward_function = forward_kernel.compile(params.get('config', {}), self._mask, jit).forward
        self._load_compile_kernel(forward_kernel)
        self.ready = True

    def forward(self, *args):
        '''Forward function. Calls the corresponding dense operator if not built.'''
        if self.ready:
            return self._sparse_forward(*args)
        else:
            warnings.warn('the sparse module is not compiled, using the dense module to forward')
            return self._raw_module.forward(*args)

    def _get_forward_kernel(self, impl: str):
        '''Get the sparse forward kernel using the specified implementation.'''
        impl_id = impl.strip().lower()
        impls = self._possible_implementations()
        if impl_id not in impls:
            raise ValueError(f'invalid implementation: {impl}')
        if self._forward_kernel is None:
            self._forward_kernel = self._create_forward_kernel(impls[impl_id])
        return self._forward_kernel

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
                search_space_cfg = TunableItemCfg('choice', [
                    {'implement': 'openai'},
                    {
                        'implement': 'sparta',
                        'config':{
                            'BLOCK_SIZE_M_VALUE': TunableItemCfg('choice', [32, 64]),
                            'BLOCK_SIZE_K_VALUE': TunableItemCfg('choice', [32, 64]),
                            'BLOCK_SIZE_N_VALUE': TunableItemCfg('choice', [32, 64]),
                            'THREAD_SIZE_M_VALUE': TunableItemCfg('choice', [4]),
                            'THREAD_SIZE_K_VALUE': TunableItemCfg('choice', [4]),
                            'THREAD_SIZE_N_VALUE': TunableItemCfg('choice', [4]),
                        }
                    },
                ])
                sparse_linear.set_search_space(search_space_cfg)

                # Tune the sparse linear layer
                sparta.tune(sparse_linear, sample_inputs=[torch.rand((512, 1024))])

        Args:
            search_space (dict): Key is the tuning algorithm, value is a dictionary whose keys are
                tunable parameters and values are lists of possible values.
        '''
        if search_space is None:
            search_space = TunableItemCfg('choice', [dict(implement=k,config=v.get_search_space()) for k,v in self._possible_implementations.items()])
        self._tuner = Tunable(search_space_cfg=search_space)

    def tune(self, sample_inputs: List[torch.Tensor], algo: str = 'grid', max_trials: int = -1):
        '''Go through all possible implementations and corresponding search spaces,
        find the best implementation and the best configuration.

        Args:
            sample_inputs (list[torch.Tensor]): Sample input tensors to determine shape
                parameters which cannot be tuned.
            algo (str): The tuning algorithm. Only grid search is supported now.
            max_trials (int): Maximum trial number. Negative value means infinity.

        Returns:
            tuple: The first value is the best implementation, the second value is the best config.
                Return (None, None) if all trials fail.
        '''
        max_trials = np.Inf if max_trials < 0 else max_trials
        if algo.strip().lower() == 'grid':
            tuner_class = tuners.GridSearchTunner
        else:
            raise ValueError(f'unsupported tuning algorithm: {algo}')
        shape, inputs = self._read_sample_inputs(*sample_inputs)
        best_impl = None
        best_cfg = None
        best_latency = np.Inf
        print(f'==================== Tuning ====================')
        for implementation, kernel_class in self._possible_implementations().items():
            kernel = self._create_forward_kernel(kernel_class)
            if implementation in self._custom_search_space:
                kernel.set_search_space(self._custom_search_space[implementation])
            space = kernel.get_search_space()
            tuner = tuner_class(space)
            space_size = round(np.exp(np.sum(np.log([len(s) for s in space.values()]))))
            print(f'----- Implementation: {implementation}; Search space: {space_size} -----')
            impl_best_cfg = None
            impl_best_latency = np.Inf
            num = 0
            for cfg in tuner._configs():
                num += 1
                print(f'#{num}: {", ".join([f"{k}={v}" for k, v in cfg.items()])}')
                try:
                    latency = kernel.test(dict(shape, **cfg), mask=self._mask, inputs=inputs)
                except AssertionError:
                    print(f'Invalid config')
                    continue
                except subprocess.SubprocessError:
                    print(f'An error occured')
                    continue
                if latency < impl_best_latency:
                    impl_best_cfg = cfg
                    impl_best_latency = latency
                print(f'Latency: {latency} ms')
                if num >= max_trials:
                    break
            if impl_best_latency < best_latency:
                best_latency = impl_best_latency
                best_cfg = impl_best_cfg
                best_impl = implementation
        if best_impl is not None and best_cfg is not None:
            best_cfg.update(shape)
            print(f'Best implementation: {best_impl}')
            print(f'Best config: {best_cfg}')
        else:
            print('All configs test failed')
        print(f'================================================')
        return best_impl, best_cfg
