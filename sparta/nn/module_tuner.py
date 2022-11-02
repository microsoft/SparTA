# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import sys
import math
import logging
import warnings
import itertools
import subprocess
from typing import Any, List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import torch

from sparta.common.tuning import Tunable, TunableItemCfg
from sparta.specializer import OperatorBase


_logger = logging.Logger(__name__)


# class TuningContextBase(object):

#     def __init__(self):
#         self.best_latency = math.inf
#         self.best_params: Optional[Dict[str, Any]] = None
#         self._search_space: Dict[str, TunableItemCfg] = {}
#         self._children: List[TuningContextBase] = []

#     def set_param(self, key: str, val: Any):
#         self._search_space[key] = TunableItemCfg('choice', [val])

#     def tune(self):
#         param_keys = list(self._search_space.keys())
#         param_spaces = [cfg._value for cfg in self._search_space.values()]
#         for param_vals in itertools.product(*param_spaces):
#             latency = 0
#             params: Dict[str, Any] = {}
#             for child_ctx in self._children:
#                 for key, val in zip(param_keys, param_vals):
#                     child_ctx.set_param(key, val)
#                 child_ctx.tune()
#                 if child_ctx.best_params is None:
#                     latency = math.inf
#                     break
#                 else:
#                     latency += child_ctx.best_latency
#                     params = dict(**params, **child_ctx.best_params)
#             if latency < self.best_latency:
#                 self.best_latency = latency
#                 self.best_params = params


# class OperatorTuningContext(TuningContextBase):

#     def __init__(self, op: OperatorBase):
#         super().__init__()
#         a


class SpaceGroup(object):

    def __init__(self):
        self._impl: Dict[str, str] = {}
        self._upper_params: List[Dict[str, str]] = []
        self._upper_space: List[List[Any]] = []
        self._search_space: Dict[str, Dict[str, TunableItemCfg]] = {}

    def add_kernel(self, kernel_name: str, kernel_impl: str):
        self._search_space[kernel_name] = {}
        self._impl[kernel_name] = kernel_impl

    def add_upper_param(self, param_names: List[str], search_space: List[Any]):
        self._upper_params.append({p.split(';')[0]: p for p in param_names})
        self._upper_space.append(search_space)

    def add_lower_param(self, param_name: str, search_space: TunableItemCfg):
        kernel_name = param_name.split(';')[0]
        self._search_space[kernel_name][param_name] = search_space

    def space_size(self) -> int:
        upper_space_size = np.prod([len(x) for x in self._upper_space])
        lower_space_size = np.sum([
            np.prod([len(x._value) for x in s.values()])
            for s in self._search_space.values()
        ])
        return int(upper_space_size * lower_space_size)

    def tune(
        self, op: OperatorBase, sample_inputs: List[torch.Tensor], sample_grad: Optional[torch.Tensor],
        algo: str, max_trials: int, tester_kw: Dict, build_kw: Dict, tuner_kw: Dict, verbose: bool
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        upper_space_size = np.prod([len(x) for x in self._upper_space])
        lower_space_size = {
            kernel_name: np.prod([len(x._value) for x in space.values()])
            for kernel_name, space in self._search_space.items()
        }
        trail_unit = float(max_trials) / upper_space_size / np.sum(list(lower_space_size.values()))
        lower_max_trails = {
            kernel_name: math.ceil(space_size * trail_unit)
            for kernel_name, space_size in lower_space_size.items()
        }

        from nni import NoMoreTrialError
        best_latency = math.inf
        best_params = None
        for upper_config in itertools.product(*self._upper_space):
            print('----------------------------------------')
            print(f'upper params: {upper_config}')
            print('----------------------------------------')
            for kernel_name, impl in self._impl.items():
                kernel_space = self._search_space[kernel_name]
                kernel_space['_impl'] = TunableItemCfg('choice', [f'{kernel_name}={impl}'])
                for param_map, upper_param_val in zip(self._upper_params, upper_config):
                    kernel_space[param_map[kernel_name]] = TunableItemCfg('choice', [upper_param_val])
            try:
                op.build(
                    params=dict(
                        _impl=';'.join([
                            f'{kernel_name}={impl}'
                            for kernel_name, impl in self._impl.items()
                        ]),
                        **{
                            lower_param_name: lower_param_cfg._value[0]
                            for kernel_space in self._search_space.values()
                            for lower_param_name, lower_param_cfg in kernel_space.items()
                            if lower_param_name != '_impl'
                        },
                    ),
                    sample_inputs=sample_inputs
                )
            except AssertionError:
                continue
            op_best_latency = 0
            op_best_params = {}
            op_tune_failed = False
            for kernel_name, impl in self._impl.items():
                print(f'kernel: {kernel_name}, max trials = {lower_max_trails[kernel_name]}')
                # for param_map, upper_param_val in zip(self._upper_params, upper_config):
                #     kernel_space[param_map[kernel_name]] = TunableItemCfg('choice', [upper_param_val])
                tuner = Tunable.create_tuner(algo, self._search_space[kernel_name], tuner_kw)
                kernel_best_latency = math.inf
                kernel_best_params = None
                for i in range(lower_max_trails[kernel_name]):
                    try:
                        params = tuner.generate_parameters(i)
                    except NoMoreTrialError:
                        break
                    try:
                        op.build(params, sample_inputs)
                        latency = op.test(
                            kernels=[kernel_name],
                            sample_inputs=sample_inputs,
                            sample_grad=sample_grad,
                            **tester_kw
                        )[kernel_name]
                    except AssertionError:
                        print(f'params:{params} -> invalid config')
                        continue
                    # except subprocess.SubprocessError:
                    #     print(f'An error occured')
                    #     continue
                    print(f'params:{params} -> latency: {latency}')
                    tuner.receive_trial_result(i, params, latency)
                    if latency < kernel_best_latency:
                        kernel_best_latency = latency
                        kernel_best_params = {k: v for k, v in params.items() if k != '_impl'}
                tuner.trial_end(i, True)
                if kernel_best_params is None:
                    op_tune_failed = True
                    break
                else:
                    op_best_latency += kernel_best_latency
                    op_best_params = dict(**op_best_params, **kernel_best_params)
            if op_tune_failed:
                continue
            if op_best_latency < best_latency:
                best_latency = op_best_latency
                best_params = op_best_params
        return best_latency, best_params


def tune_sparse_operator(
    op: OperatorBase, sample_inputs: List[torch.Tensor], sample_grad: Optional[torch.Tensor] = None,
    backward_weight: float = 0.0, algo: str = 'grid', max_trials: int = sys.maxsize,
    tester_kw: Dict = None, build_kw: Dict = None, tuner_kw: Dict = None, verbose: bool = False
):
    best_params = None
    best_latency = math.inf
    for impl, impl_space in op.get_search_space(backward_weight > 0).items():
        print('########################################')
        print(f'impl: {impl}')
        print('########################################')
        kernel_impls: Dict[str, str] = {}
        kernel_groups: Dict[str, int] = {}
        for i, kernel_impl_str in enumerate(impl.split(';')):
            kernel_name, kernel_impl = kernel_impl_str.split('=')
            kernel_impls[kernel_name] = kernel_impl
            kernel_groups[kernel_name] = i
        space: Dict[str, TunableItemCfg] = impl_space['_space']
        params = set(space.keys())
        upper_params: List[Tuple[List[str], List]] = []
        for condition in impl_space['_conditions']:
            group_idx = None
            fixed_value = None
            connected_params = []
            for param in condition:
                if param in params:
                    connected_params.append(param)
                    params.remove(param)
                    kernel_name = param.split(';')[0]
                    if group_idx is None:
                        group_idx = kernel_groups[kernel_name]
                    else:
                        kernel_groups[kernel_name] = group_idx
                else:
                    fixed_value = param
            if fixed_value is None:
                spaces = [set(space[p]._value) for p in connected_params]
                connected_space = sorted(list(spaces[0].intersection(*spaces[1:])))
            else:
                connected_space = [fixed_value]
            upper_params.append((connected_params, connected_space))
        space_groups: Dict[int, SpaceGroup] = {}
        for kernel_name, group_idx in kernel_groups.items():
            if group_idx not in space_groups:
                space_groups[group_idx] = SpaceGroup()
            space_groups[group_idx].add_kernel(kernel_name, kernel_impls[kernel_name])
        for param_list, search_space in upper_params:
            kernel_name = param_list[0].split(';')[0]
            space_group = space_groups[kernel_groups[kernel_name]]
            space_group.add_upper_param(param_list, search_space)
        for param_name in params:
            kernel_name = param_name.split(';')[0]
            space_group = space_groups[kernel_groups[kernel_name]]
            space_group.add_lower_param(param_name, space[param_name])
        group_space_size = [space_group.space_size() for space_group in space_groups.values()]
        trial_unit = float(max_trials) / np.sum(group_space_size)
        impl_failed = False
        impl_best_latency = 0
        impl_best_params = {}
        for space_group, space_size in zip(space_groups.values(), group_space_size):
            group_best_latency, group_best_params = space_group.tune(
                op=op,
                sample_inputs=sample_inputs,
                sample_grad=sample_grad,
                algo=algo,
                max_trials=math.ceil(space_size * trial_unit),
                tester_kw=tester_kw or {},
                build_kw=build_kw or {},
                tuner_kw=tuner_kw or {},
                verbose=verbose,
            )
            if group_best_params is None:
                impl_failed = True
                break
            else:
                impl_best_latency += group_best_latency
                impl_best_params = dict(**impl_best_params, **group_best_params)
        if impl_failed:
            continue
        if impl_best_latency < best_latency:
            best_latency = impl_best_latency
            best_params = dict(
                _impl=';'.join([
                    f'{kernel_name}={kernel_impl}'
                    for kernel_name, kernel_impl in kernel_impls.items()
                ]),
                **impl_best_params
            )
    if best_params is None:
        print('All trails failed')
    else:
        print(f'Best params: {best_params}')
        op.build(best_params, sample_inputs=sample_inputs)


def tune_combined_module(
    module: torch.nn.Module, sample_inputs: List[torch.Tensor],
    algo: str = 'grid', max_trials: int = sys.maxsize, tester_kw: Dict = None,
    build_kw: Dict = None, tuner_kw: Dict = None, verbose: bool = False
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
        module_dict: Dict[str, torch.nn.Module] = field(default_factory=dict)
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
                latency += module.tester(params[name], sample_inputs=ctx.input_dict[name], **tester_kw)
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
