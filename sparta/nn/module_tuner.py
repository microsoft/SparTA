# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import sys
import random
import logging
import warnings
from typing import Any, List, Dict, Callable, Iterator, Optional
from dataclasses import dataclass, field

import torch
import numpy as np

from sparta.common.tuning import Tunable, TunableItemCfg
from sparta.specializer import OperatorBase


_logger = logging.Logger(__name__)
_handler = logging.StreamHandler()
_logger.addHandler(_handler)


# def tune_combined_module(
#     module: torch.nn.Module, sample_inputs: List[torch.Tensor], sample_grads: List[torch.Tensor],
#     algo: str = 'grid', max_trials: int = sys.maxsize, backward_weight: float = 0,
#     tester_kw: Dict = None, build_kw: Dict = None, tuner_kw: Dict = None, verbose: bool = False
# ):
#     """Find, tune and build all sparse operators in the model.

#     Args:
#         module (torch.nn.Module): A PyTorch module that contains one or more sparse sub-modules.
#         sample_inputs (List[torch.Tensor]): Sample input tensors to determine shape parameters.
#         algo: (str, optional): The algorithm to search the best parameters. Defaults to 'grid'.
#         max_trials: (int, optional): The maximum number of trials to run. Defaults to sys.maxsize.
#         tester_kw: (Dict, optional): The keyword arguments for the tester. Defaults to None.
#         build_kw: (Dict, optional): The keyword arguments for the builder (after tuning). Defaults to None.
#         tuner_kw: (Dict, optional): The keyword arguments for the tuner. Defaults to None.
#     """
#     from nni import NoMoreTrialError

#     @dataclass
#     class _TuningContext:
#         """Context for tuning."""
#         module_dict: Dict[str, OperatorBase] = field(default_factory=dict)
#         space_dict: Dict[str, TunableItemCfg] = field(default_factory=dict)
#         input_dict: Dict[str, list] = field(default_factory=dict)
#         best_latency: float = math.inf
#         best_params: Dict = None

#         def add(self, name, module, space, inputs):
#             """Add a module to the context."""
#             _logger.info(f'tunable operator deduced {type(module)} {name} ')
#             self.module_dict[name] = module
#             self.space_dict[name] = space
#             self.input_dict[name] = inputs

#     ctx = _TuningContext()

#     if isinstance(module, OperatorBase):
#         ctx.add('root', module, module.get_search_space(), sample_inputs)
#     else:
#         sample_inputs_dict = {}
#         for child_name, child_module in module.named_children():
#             sample_inputs_dict[child_name] = []
#             child_module.register_forward_hook(get_input_hook(sample_inputs_dict, child_name))
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             module.forward(*sample_inputs)
#         for child_name, child_module in module.named_children():
#             if isinstance(child_module, OperatorBase):
#                 ctx.add(child_name, child_module, child_module.get_search_space(), sample_inputs_dict[child_name])

#     tuner = Tunable.create_tuner(algo, ctx.space_dict, tuner_kw)
#     tester_kw = tester_kw or {}
#     for i in range(max_trials):
#         try:
#             params = tuner.generate_parameters(i)
#         except NoMoreTrialError:
#             break
#         latency = 0.0
#         try:
#             for name, module in ctx.module_dict.items():
#                 latency += module.test(
#                     params[name],
#                     sample_inputs=ctx.input_dict[name],
#                     **tester_kw
#                 )
#         except AssertionError:
#             _logger.warn(f'Invalid config')
#             continue
#         _logger.info(f'params:{params} -> latency: {latency}')
#         tuner.receive_trial_result(i, params, latency)  # TODO: add status here
#         if latency < ctx.best_latency:
#             ctx.best_latency = latency
#             ctx.best_params = params
#     tuner.trial_end(i, True)

#     build_kw = build_kw or {}
#     for name, module in ctx.module_dict.items():
#         module.build(ctx.best_params[name], sample_inputs=ctx.input_dict[name], **build_kw)
#     return ctx.best_params


class Tuner(object):

    def __init__(
        self, search_space: Dict[Any, TunableItemCfg],
        eval_func: Callable[[int, Dict[Any, Any]], float],
        max_trials: int = sys.maxsize
    ):
        self._search_space = search_space
        self._eval_func = eval_func
        space_shape = [len(param_space._value) for param_space in search_space.values()]
        space_size = int(np.prod(space_shape))
        self._max_trials = min(max_trials, space_size)
        self.best_result = np.inf
        self.best_config = None

    @abc.abstractmethod
    def next_config(self) -> Iterator[Dict[str, Any]]:
        """Yields the next config."""

    def tune(self):
        for i, config in zip(range(self._max_trials), self.next_config()):
            result = self._eval_func(i, config)
            if result < self.best_result:
                self.best_result = result
                self.best_config = config


class RandomSearchTuner(Tuner):

    def next_config(self):
        while True:
            yield {
                param_name: random.choice(param_space._value)
                for param_name, param_space in self._search_space.items()
            }


class GridSearchTuner(Tuner):

    def next_config(self):
        if len(self._search_space) == 0:
            yield {}
        else:
            param_names = []
            param_idxs = []
            param_space_sizes = []
            for param_name, param_space in self._search_space.items():
                param_names.append(param_name)
                param_space_sizes.append(len(param_space._value))
                param_idxs.append(0)
            while param_idxs[0] < param_space_sizes[0]:
                yield {
                    param_name: self._search_space[param_name]._value[param_idx]
                    for param_idx, param_name in zip(param_idxs, param_names)
                }
                k = len(param_idxs) - 1
                param_idxs[k] += 1
                while param_idxs[k] == param_space_sizes[k] and k > 0:
                    param_idxs[k - 1] += 1
                    param_idxs[k] = 0
                    k -= 1


def tune_sparse_module(
    module: OperatorBase,
    name: str,
    sample_inputs: List[torch.Tensor],
    sample_grads: Optional[List[torch.Tensor]] = None,
    algo: str = 'grid',
    max_trials: int = sys.maxsize,
    backward_weight: float = 0,
    debug_func: Optional[Callable[[int, Dict[Any, Any]], float]] = None,
):
    if algo.startswith('grid'):
        tuner_type = GridSearchTuner
    elif algo.startswith('rand'):
        tuner_type = RandomSearchTuner
    else:
        raise ValueError(f'unsupported tuner algorithm "{algo}"')

    module.set_sample_inputs(sample_inputs, sample_grads)
    search_space = module.get_search_space(backward_weight > 0)
    connections = module.get_connections(backward_weight > 0)
    upper_space = [
        set.intersection(*[
            set.union(*[
                set(impl_space[param_name]._value)
                for impl_space in search_space[kernel_name].values()
            ])
            for kernel_name, param_name in connection.items()
        ])
        for connection in connections
    ]
    upper_space_shape = [len(param_space) for param_space in upper_space]
    upper_space_size = int(np.prod(upper_space_shape))
    upper_space = {
        i: TunableItemCfg('choice', list(param_space))
        for i, param_space in enumerate(upper_space)
    }

    lower_params_cache = {}

    def lower_search(upper_idx: int, upper_params: Dict[Any, Any]):
        _logger.info(f'[{name}][Upper Search Space] #{upper_idx}: {list(upper_params.values())}')
        lower_params = {}
        lower_best_latency = 0
        for kernel_name, kernel in module.get_kernel_placeholders(backward_weight > 0).items():
            _logger.info(f'[{name}][Kernel: {kernel_name}]')
            kernel_max_trials = int(np.ceil(max_trials / upper_space_size))
            kernel_best_params = None
            kernel_best_latency = np.inf
            kernel_weight = backward_weight if kernel_name.startswith('backward') else 1
            fixed_params = {
                connections[i][kernel_name]: val
                for i, val in upper_params.items()
            }
            for impl, kernel_space in kernel.get_search_space(fixed_params).items():
                def try_params(lower_idx: int, params: Dict[Any, Any]):
                    try:
                        kernel.build(impl, params)
                        latency = kernel.test()
                    except AssertionError:
                        latency = np.inf
                    _logger.info(f'{impl} #{lower_idx}: {list(params.values())} => {latency} ms')
                    return latency
                func = try_params if debug_func is None else debug_func
                kernel_tuner = tuner_type(kernel_space, func, kernel_max_trials)
                kernel_tuner.tune()
                if kernel_tuner.best_result < kernel_best_latency:
                    kernel_best_params = dict(_impl=impl, **kernel_tuner.best_config)
                    kernel_best_latency = kernel_tuner.best_result
            lower_params[kernel_name] = kernel_best_params
            lower_best_latency += kernel_best_latency * kernel_weight
        lower_params_cache[str(upper_params)] = lower_params
        return lower_best_latency

    tuner = tuner_type(upper_space, lower_search, upper_space_size)
    tuner.tune()
    _logger.info(f'[{name}] Tuning completed.')
    if tuner.best_config is None:
        _logger.warn(f'[{name}] All trials failed.')
        return None
    else:
        best_config = lower_params_cache[str(tuner.best_config)]
        _logger.info(f'[{name}] Best config:\n{best_config}')
        module.build(best_config, sample_inputs)
        return best_config


def tune_combined_module(
    module: torch.nn.Module,
    sample_inputs: List[torch.Tensor],
    sample_grads: Optional[List[torch.Tensor]] = None,
    algo: str = 'grid',
    max_trials: int = sys.maxsize,
    backward_weight: float = 0,
    debug: bool = False,
    debug_func: Optional[Callable] = None,
):
    sample_inputs_dict = {'root': sample_inputs}
    sample_grads_dict = {'root': sample_grads}

    def register_hooks(op: OperatorBase, name: str):
        op.register_forward_hook(get_input_hook(sample_inputs_dict, name))
        op.register_backward_hook(get_grad_hook(sample_grads_dict, name))

    iter_sparse_modules(module, 'root', register_hooks)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if backward_weight > 0:
            for x in sample_inputs:
                x.requires_grad = True
        outputs = module.forward(*sample_inputs)
        if backward_weight > 0:
            if type(outputs) is not tuple:
                outputs = (outputs, )
            for output, sample_grad in zip(outputs, sample_grads):
                if type(output) is torch.Tensor:
                    output.backward(sample_grad)
            for x in sample_inputs:
                x.requires_grad = False

    best_configs = {}

    if debug:
        _logger.setLevel(logging.DEBUG)
    else:
        _logger.setLevel(logging.WARNING)

    def tune(op: OperatorBase, name: str):
        best_configs[name] = tune_sparse_module(
            module=op,
            name=name,
            sample_inputs=sample_inputs_dict[name],
            sample_grads=sample_grads_dict[name] if name in sample_grads_dict else None,
            algo=algo,
            max_trials=max_trials,
            backward_weight=backward_weight,
            debug_func=debug_func,
        )

    iter_sparse_modules(module, 'root', tune)
    return best_configs


def iter_sparse_modules(
    module: torch.nn.Module,
    module_name: str,
    func: Callable[[OperatorBase, str], None],
):
    if isinstance(module, OperatorBase):
        func(module, module_name)
        return
    for child_name, child_module in module.named_children():
        iter_sparse_modules(child_module, f'{module_name}/{child_name}', func)


def get_input_hook(input_dict: Dict[str, List], module_name: str):
    """Create a hook to capture the input tensor(s) and save to a dictionary

    Args:
        input_dict (Dict): The dictionary to save input tensor(s).
        module_name (str): Module name as the index of the input dictionary.

    Returns:
        Callable: The input hook function.
    """
    def input_hook(module, fea_in, fea_out):
        input_dict[module_name] = list(fea_in)

    return input_hook


def get_grad_hook(grad_dict: Dict[str, List], module_name: str):
    """Create a hook to capture the grad tensor(s) and save to a dictionary

    Args:
        grad_dict (Dict): The dictionary to save grad tensor(s).
        module_name (str): Module name as the index of the grad dictionary.

    Returns:
        Callable: The grad hook function.
    """
    def grad_hook(module, grad_input, grad_output):
        grad_dict[module_name] = list(grad_output)

    return grad_hook
