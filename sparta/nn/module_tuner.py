# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import logging
import warnings
import subprocess
from typing import List, Dict, Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from sparta.common.tuning import Tunable, TunableItemCfg
from sparta.specializer import OperatorBase


_logger = logging.Logger(__name__)


def tune_combined_module(
    module: torch.nn.Module, sample_inputs: List[torch.Tensor], sample_grads: List[torch.Tensor],
    algo: str = 'grid', max_trials: int = sys.maxsize, backward_weight: float = 0,
    tester_kw: Dict = None, build_kw: Dict = None, tuner_kw: Dict = None, verbose: bool = False
):
    '''Find, tune and build all sparse operators in the model.

    Args:
        module (torch.nn.Module): A PyTorch module that contains one or more sparse sub-modules.
        sample_inputs (List[torch.Tensor]): Sample input tensors to determine shape parameters.
        algo: (str, optional): The algorithm to search the best parameters. Defaults to 'grid'.
        max_trials: (int, optional): The maximum number of trials to run. Defaults to sys.maxsize.
        tester_kw: (Dict, optional): The keyword arguments for the tester. Defaults to None.
        build_kw: (Dict, optional): The keyword arguments for the builder (after tuning). Defaults to None.
        tuner_kw: (Dict, optional): The keyword arguments for the tuner. Defaults to None.
    '''
    from nni import NoMoreTrialError

    @dataclass
    class _TuningContext:
        '''Context for tuning.'''
        module_dict: Dict[str, OperatorBase] = field(default_factory=dict)
        space_dict: Dict[str, TunableItemCfg] = field(default_factory=dict)
        input_dict: Dict[str, list] = field(default_factory=dict)
        best_latency: float = np.inf
        best_params: Dict = None

        def add(self, name, module, space, inputs):
            '''Add a module to the context.'''
            _logger.info(f'tunable operator deduced {type(module)} {name} ')
            self.module_dict[name] = module
            self.space_dict[name] = space
            self.input_dict[name] = inputs

    ctx = _TuningContext()

    if isinstance(module, OperatorBase):
        ctx.add('root', module, module.get_search_space(), sample_inputs)
    else:
        sample_inputs_dict = {}
        for child_name, child_module in module.named_children():
            sample_inputs_dict[child_name] = []
            child_module.register_forward_hook(get_input_hook(sample_inputs_dict, child_name))
            # child_module.register_backward_hook
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            module.forward(*sample_inputs)
        for child_name, child_module in module.named_children():
            if isinstance(child_module, OperatorBase):
                ctx.add(child_name, child_module, child_module.get_search_space(), sample_inputs_dict[child_name])

    tuner = Tunable.create_tuner(algo, ctx.space_dict, tuner_kw)
    tester_kw = tester_kw or {}
    for i in range(max_trials):
        try:
            params = tuner.generate_parameters(i)
        except NoMoreTrialError:
            break
        latency = 0.0
        try:
            for name, module in ctx.module_dict.items():
                latency += module.test(
                    params[name],
                    sample_inputs=ctx.input_dict[name],
                    sample_grad=a,
                    **tester_kw
                )
        except AssertionError:
            _logger.warn(f'Invalid config')
            continue
        except subprocess.SubprocessError:
            _logger.warn(f'An error occured')
            continue
        _logger.info(f'params:{params} -> latency: {latency}')
        tuner.receive_trial_result(i, params, latency)  # TODO: add status here
        if latency < ctx.best_latency:
            ctx.best_latency = latency
            ctx.best_params = params
    tuner.trial_end(i, True)

    build_kw = build_kw or {}
    for name, module in ctx.module_dict.items():
        module.build(ctx.best_params[name], sample_inputs=ctx.input_dict[name], **build_kw)
    return ctx.best_params


def iter_sparse_modules(
    module: torch.nn.Module, module_name: str,
    func: Callable[[OperatorBase, str], None]
):
    if isinstance(module, OperatorBase):
        func(module, module_name)
        return
    for chile_name, child_module in module.named_children():
        iter_sparse_modules(child_module, chile_name, func)


def get_input_hook(input_dict: Dict[str, list], module_name: str):
    '''Create a hook to capture the input tensor(s) and save to a dictionary

    Args:
        input_dict (Dict): The dictionary to save input tensor(s).
        module_name (str): Module name as the index of the input dictionary.

    Returns:
        Callable: The input hook function.
    '''
    def input_hook(module, fea_in, fea_out):
        input_dict[module_name] = list(fea_in)

    return input_hook


def get_grad_hook(grad_dict: Dict[str, list], module_name: str):
    '''Create a hook to capture the grad tensor(s) and save to a dictionary

    Args:
        grad_dict (Dict): The dictionary to save grad tensor(s).
        module_name (str): Module name as the index of the grad dictionary.

    Returns:
        Callable: The grad hook function.
    '''
    def grad_hook(module, fea_in, fea_out):
        grad_dict[module_name] = list(fea_in)

    return grad_hook
