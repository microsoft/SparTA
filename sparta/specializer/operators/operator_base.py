# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import sys
import warnings
import subprocess
from typing import Type, Tuple, List, Dict

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
        base_class (Type[torch.nn.Module]): Class of the dense operator.
    '''

    def __init__(self, raw_module: torch.nn.Module, base_class: Type[torch.nn.Module], name: str = None):
        if type(raw_module) is not base_class:
            raise ValueError(f'expected a {base_class} module')
        super().__init__()
        self._name = name or get_uname()
        self._raw_module = raw_module
        self._forward_function = None
        self._tuner = None
        self._mask = None
        self.ready = False
        self._possible_implementations = {}

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
        self._tuner = Tunable(search_space_cfg=search_space, name=self._name)

    def tune(self, sample_inputs: List[torch.Tensor], algo: str = 'grid', max_trials: int = sys.maxsize):
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
        from nni import NoMoreTrialError

        def _split_params(p: Dict):
            implement, cfg = params[self._name]['_name'], params[self._name]
            return implement, cfg
        if self._tuner is None:  # use default search space
            self.set_search_space()
        tuner = self._tuner.create_tuner(algo)
        shape, inputs = self._read_sample_inputs(*sample_inputs)
        best_latency = np.Inf
        print(f'==================== Tuning ====================')

        for i in range(max_trials):
            try:
                params = tuner.generate_parameters(i)
            except NoMoreTrialError:
                break
            print(f'#{i}: {params}')
            implement, cfg = _split_params(params)
            # For JIT test:
            # self.build(params, sample_inputs=sample_inputs)
            kernel = self._possible_implementations[implement]
            try:
                latency = kernel.test(dict(shape, **cfg), mask=self._mask, inputs=inputs)
            except AssertionError:
                print(f'Invalid config')
                continue
            except subprocess.SubprocessError:
                print(f'An error occured')
                continue
            print(f'Latency: {latency} ms')
            tuner.receive_trial_result(i, params, latency)  # TODO: add status here
            if latency < best_latency:
                best_latency = latency
                best_params = params[self._name]
        tuner.trial_end(i, True)
        if best_params is None:
            print('All configs test failed')
        else:
            print(f'Minimum latency: {best_latency} ms')
            print(f'Best config: {best_params}')
        return best_params
