# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import warnings
import subprocess
from typing import Optional

import torch
import numpy as np

from sparta.specializer import kernels, tuners


class OperatorBase(torch.nn.Module):
    '''Base class of sparse operators.

    Args:
        raw_module (torch.nn.Module): The corresponding dense operator.
        base_class (type[torch.nn.Module]): Class of the dense operator.
    '''

    def __init__(self, raw_module: torch.nn.Module, base_class: type[torch.nn.Module]):
        if type(raw_module) is not base_class:
            raise ValueError(f'expected a {base_class} module')
        super().__init__()
        self._raw_module = raw_module
        self._forward_kernel = None
        self._forward_function = None
        self._custom_search_space = {}
        self._mask = None
        self.ready = False

    def build(self, impl: str, config: dict, jit: bool = True):
        '''Build the sparse kernel using the specified implementation and configs.

        Args:
            impl (str): Implementation. Can be extracted from the tuning result.
            config (str): Kernel config. Can be extracted from the tuning result.
            jit (bool): Determine whether to build the kernel using JIT mode.
        '''
        forward_kernel = self._get_forward_kernel(impl)
        self._forward_function = forward_kernel.compile(config, self._mask, jit).forward
        self._set_parameters(forward_kernel)
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
    def _set_parameters(self, forward_kernel: kernels.KernelBase):
        '''Set PyTorch module parameters according to the dense operator.

        Args:
            forward_kernel (kernels.KernelBase): The forward kernel object
                which provides the sparsify function.
        '''

    @abc.abstractmethod
    def _possible_implementations(self) -> dict[str, type[kernels.KernelBase]]:
        '''Get possible implementations.

        Returns:
            dict: Key is the implementation name, value is the corresponding kernel class.
        '''

    @abc.abstractmethod
    def _create_forward_kernel(self, kernel_class: type[kernels.KernelBase]) -> kernels.KernelBase:
        '''Instantiate a forward kernel object using the specified kernel class.'''

    @abc.abstractmethod
    def _read_sample_inputs(self, *args) -> tuple[dict, dict]:
        '''Read shape config and convert sample inputs to test inputs.'''

    def set_search_space(self, search_space: dict[str, dict[str, list]]):
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
                sparse_softmax.set_search_space({
                    'sparta': {
                        'BLOCK_SIZE_M_VALUE': [32, 64],
                        'BLOCK_SIZE_K_VALUE': [32, 64],
                        'BLOCK_SIZE_N_VALUE': [32, 64],
                        'THREAD_SIZE_M_VALUE': [4],
                        'THREAD_SIZE_K_VALUE': [4],
                        'THREAD_SIZE_N_VALUE': [4],
                    }
                })

                # Tune the sparse linear layer
                sparta.tune(sparse_linear, sample_inputs=[torch.rand((512, 1024))])

        Args:
            search_space (dict): Key is the tuning algorithm, value is a dictionary whose keys are
                tunable parameters and values are lists of possible values.
        '''
        self._custom_search_space = search_space

    def tune(self, sample_inputs: list[torch.Tensor], algo: str = 'grid', max_trials: int = -1):
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
                    latency = kernel.test(shape | cfg, mask=self._mask, inputs=inputs)
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
            best_cfg |= shape
            print(f'Best implementation: {best_impl}')
            print(f'Best config: {best_cfg}')
        else:
            print('All configs test failed')
        print(f'================================================')
        return best_impl, best_cfg
