# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import logging
import warnings
from typing import Any, List, Dict, Callable, Optional

import torch
import numpy as np

from sparta.tuning import TunableItemCfg, GridSearchTuner, RandomSearchTuner
from sparta.operators import SparseOperator


_logger = logging.Logger(__name__)
_handler = logging.StreamHandler()
_logger.addHandler(_handler)


def tune_sparse_module(
    module: SparseOperator,
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
                kernel.select_impl(impl)
                def try_params(lower_idx: int, params: Dict[Any, Any]):
                    try:
                        kernel.build(params)
                        latency = kernel.profile()
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
    if debug_func is None:
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
    verbose: bool = False,
    debug_func: Optional[Callable] = None,
):
    """Tune all sparse child operators in a combined module.
    The function will search out all sparse operators, get corresponding inputs and gradients using forward and
    backward hooks, and tune each sparse operator with given parameters.

    Args:
        module (torch.nn.Module): It can be either a SparTA sparse operator or a combined PyTorch
            module including one or more sparse operators.
        sample_inputs (List[torch.Tensor]): Sample inputs of the module.
        sample_grads (Optional[List[torch.Tensor]]): Sample gradients of the module.
        algo (str): Tuning algorithm. Only "grid" (grid search) and "rand" (random search) are supported now.
        max_trials (int): The maximum trial number the tuner will take.
        backward_weight (float): The weight for backward latency. 0 means backward is not considered.
        verbose (bool): Whether to log verbose infomation.
        debug_func (Optional[Callable]): Debug function to call on each trial.

    Returns:
        Dict[str, Dict[str, Any]]: Best configs of all sparse operators in the combined module.
    """
    sample_inputs_dict = {'root': sample_inputs}
    sample_grads_dict = {'root': sample_grads}
    hook_handlers = []

    def register_hooks(op: SparseOperator, name: str):
        hook_handlers.append(op.register_forward_hook(get_input_hook(sample_inputs_dict, name)))
        hook_handlers.append(op.register_full_backward_hook(get_grad_hook(sample_grads_dict, name)))

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

    for handler in hook_handlers:
        handler.remove()

    best_configs = {}

    if verbose:
        _logger.setLevel(logging.DEBUG)
    else:
        _logger.setLevel(logging.WARNING)

    def tune(op: SparseOperator, name: str):
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


def build_combined_module(
    module: torch.nn.Module,
    sample_inputs: List[torch.Tensor],
    configs: Dict[str, Dict[str, Any]],
):
    """Build all sparse child operators in a combined module.
    The function will search out all sparse operators, get corresponding inputs using forward hooks, and build each
    sparse operator with given config.

    Args:
        module (torch.nn.Module): It can be either a SparTA sparse operator or a combined PyTorch
            module including one or more sparse operators.
        sample_inputs (List[torch.Tensor]): Sample inputs of the module.
        configs (Dict[str, Dict[str, Any]]): Best configs of all sparse operators in the combined module.
    """
    sample_inputs_dict = {'root': sample_inputs}
    hook_handlers = []

    def register_hooks(op: SparseOperator, name: str):
        hook_handlers.append(op.register_forward_hook(get_input_hook(sample_inputs_dict, name)))

    iter_sparse_modules(module, 'root', register_hooks)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        outputs = module.forward(*sample_inputs)

    for handler in hook_handlers:
        handler.remove()

    def build(op: SparseOperator, name: str):
        op.build(configs[name], sample_inputs=sample_inputs_dict[name])

    iter_sparse_modules(module, 'root', build)


def iter_sparse_modules(
    module: torch.nn.Module,
    module_name: str,
    func: Callable[[SparseOperator, str], None],
):
    if isinstance(module, SparseOperator):
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
