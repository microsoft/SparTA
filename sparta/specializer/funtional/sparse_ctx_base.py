# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Any, Dict, List, Tuple, Type, Callable, Optional

import torch

from sparta.testing.utils import test_latency
from sparta.specializer.kernels import KernelBase


class KernelPlaceholder(object):

    def __init__(
        self, name: str, cat: str, impls: Dict[str, Type[KernelBase]],
        args: Dict[str, Any], mask_map: Dict[str, str]
    ):
        self.name = name
        self.category = cat
        self._possible_kernels = {key: impl(**args) for key, impl in impls.items()}
        self._active_impl: str = None
        self._mask_map = mask_map
        self.ready = False

    def set_shape(self, *args, **kwargs):
        for kernel in self._possible_kernels.values():
            kernel.set_shape(*args, **kwargs)

    def build(self, impl: str, config: Dict[str, Any], mask: Dict[str, torch.Tensor]):
        self._active_impl = impl
        mapped_mask = {y: mask[x] for x, y in self._mask_map.items()}
        self._possible_kernels[impl].compile(config, mapped_mask)
        self.ready = True

    def get_converter(self, name: str):
        return self.active_kernel().get_converter(self._mask_map[name])

    def active_kernel(self):
        return self._possible_kernels[self._active_impl]

    def get_search_spase(self):
        return {
            impl: kernel.get_search_space()
            for impl, kernel in self._possible_kernels.items()
        }


class SparseCtxBase(object):

    def __init__(self):
        self._kernels: Dict[str, KernelPlaceholder] = {}
        self._mask: Dict[str, torch.Tensor] = {}

    @abc.abstractmethod
    def set_shape(self, *args, **kwargs):
        '''Set shape parameters.'''

    def build(
        self, config: Dict[str, Any], mask: Dict[str, torch.Tensor]
    ):
        assert '_impl' in config, 'implementation should be specified in config'
        for impl in config['_impl'].split(';'):
            kernel_name, impl_name = impl.split('=')
            filtered_config = {
                param_key.split(';')[1]: param_val
                for param_key, param_val in config.items()
                if param_key.startswith(kernel_name)
            }
            self._kernels[kernel_name].build(impl_name, filtered_config, mask)

    def get_converter(self, kernel_name: str, tensor_name: str):
        return self._kernels[kernel_name].get_converter(tensor_name)

    @abc.abstractmethod
    def get_conditions(self, impls: Dict[str, str]) -> Optional[List[List]]:
        '''Get conditions given implementation of each kernel.'''

    def get_kernels(self):
        return list(self._kernels.keys())

    @abc.abstractmethod
    def _split_graph(
        self, kernels: List[str], sample_inputs: Dict[str, torch.Tensor],
        sample_grad: Optional[torch.Tensor] = None
    ) -> Tuple[List[Callable], List[List[torch.Tensor]]]:
        '''Split the calculation graph into kernels with input tensors.'''

    def test(
        self, kernels: List[str], sample_inputs: Dict[str, torch.Tensor],
        sample_grad: Optional[torch.Tensor] = None,
        num_warmups: int = 10, num_iters: int = 10
    ):
        funcs, inputs = self._split_graph(kernels, sample_inputs, sample_grad)
        latency_dict: Dict[str, float] = {}
        for kernel_name, func, input_list in zip(kernels, funcs, inputs):
            latency_dict[kernel_name] = test_latency(
                func=func,
                inputs=input_list,
                num_warmups=num_warmups,
                num_iters=num_iters,
                cuda=True,
            )
        return latency_dict

    def _expand_search_space(self, kernels: List[str], impls: Dict[str, str]) -> Dict[str, Dict]:
        if len(kernels) == 0:
            conditions = self.get_conditions(impls)
            if conditions is None:
                return {}
            else:
                impl_str = ';'.join([f'{kernel}={impl}' for kernel, impl in impls.items()])
                return {impl_str: {'_space': {}, '_conditions': conditions}}
        return {
            impl: {
                '_space': dict(
                    **{f'{kernels[0]};{k}': v for k, v in kernel_space.items()},
                    **space['_space'],
                ),
                '_conditions': space['_conditions']
            }
            for kernel_impl, kernel_space in self._kernels[kernels[0]].get_search_spase().items()
            for impl, space in self._expand_search_space(
                kernels[1:],
                dict(**impls, **{kernels[0]: kernel_impl})
            ).items()
        }

    def get_search_space(self, backward: bool = False):
        return self._expand_search_space(
            kernels=[
                kernel_name
                for kernel_name in self._kernels.keys()
                if backward or not kernel_name.startswith('backward')
            ],
            impls={}
        )
