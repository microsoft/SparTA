# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import warnings
import subprocess
from typing import Optional, Iterable

import torch
import numpy as np

from sparta.specializer import kernels, tuners


class OperatorBase(torch.nn.Module):

    def __init__(self, implementation: str):
        super().__init__()
        self.set_implementation(implementation)
        self._forward_kernel = None
        self._forward_function = None
        self._mask = None
        self.ready = False

    def build(self, config: dict, jit: bool = True):
        forward_kernel = self._get_forward_kernel()
        self._forward_function = forward_kernel.compile(config, self._mask, jit).forward
        self._set_parameters(forward_kernel)
        self.ready = True

    def forward(self, *args):
        if self.ready:
            return self._sparse_forward(*args)
        else:
            warnings.warn('the sparse module is not compiled, using the dense module to forward')
            return self._raw_module.forward(*args)

    def set_implementation(self, implementation: str):
        if implementation.lower() in self._possible_implementations():
            self._implementation = implementation.lower()
        else:
            raise ValueError(f'invalid implementation: {implementation}')

    def _get_forward_kernel(self):
        if self._forward_kernel is None:
            kernel_class = self._possible_implementations()[self._implementation]
            self._forward_kernel = self._create_forward_kernel(kernel_class)
        return self._forward_kernel

    @abc.abstractmethod
    def _sparse_forward(self, *args):
        '''
        forward using the sparse kernel if ready
        '''

    @abc.abstractmethod
    def _set_parameters(self, forward_kernel: kernels.KernelBase):
        '''
        set PyTorch parameters
        '''

    @abc.abstractmethod
    def _possible_implementations(self) -> dict[str, type[kernels.KernelBase]]:
        '''
        get possible implementations
        '''

    @abc.abstractmethod
    def _create_forward_kernel(self, kernel_class: type[kernels.KernelBase]) -> kernels.KernelBase:
        '''
        instantiate the forward kernel
        '''

    @abc.abstractmethod
    def _read_sample_inputs(self, *args) -> tuple[dict, dict]:
        '''
        read shape config and convert sample inputs into test inputs
        '''

    def tune(
        self, sample_inputs: Iterable[torch.Tensor], tuner_type: str = 'grid',
        search_space: Optional[dict[str, dict[str, list]]] = None
    ):
        search_space = self._search_space if search_space is None else search_space
        if tuner_type == 'grid':
            tuner_class = tuners.GridSearchTunner
        else:
            raise ValueError(f'unsupported tuner: {tuner_type}')
        shape, inputs = self._read_sample_inputs(*sample_inputs)
        best_impl = None
        best_cfg = None
        best_latency = float('inf')
        print(f'==================== Tuning ====================')
        for implementation, kernel_class in self._possible_implementations().items():
            print(f'---------- Implementation: {implementation} ----------')
            kernel = self._create_forward_kernel(kernel_class)
            if implementation in search_space:
                kernel.set_search_space(search_space[implementation])
            tuner = tuner_class(kernel.get_search_space())
            impl_best_cfg = None
            impl_best_latency = float('inf')
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
            if impl_best_latency < best_latency:
                best_latency = impl_best_latency
                best_cfg = impl_best_cfg
                best_impl = implementation
        if best_impl is not None and best_cfg is not None:
            best_cfg |= shape
            print(f'Best implementation: {best_impl}')
            print(f'Best config: {", ".join([f"{k}={v}" for k, v in best_cfg.items()])}')
            self.set_implementation(best_impl)
        else:
            print('All configs test failed')
        print(f'================================================')
        return best_cfg
