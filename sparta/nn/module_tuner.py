# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import List, Dict

import torch

from sparta.specializer import OperatorBase


def tune_combined_module(module: torch.nn.Module, sample_inputs: List[torch.Tensor]):
    '''Find, tune and build all sparse operators in the model.

    Args:
        module (torch.nn.Module): A PyTorch module that contains one or more sparse sub-modules.
        sample_inputs (List[torch.Tensor]): Sample input tensors to determine shape parameters.
    '''
    if isinstance(module, OperatorBase):
        tune_sparse_module(module, sample_inputs)
    else:
        sample_inputs_dict = {}
        for child_name, child_module in module.named_children():
            sample_inputs_dict[child_name] = []
            child_module.register_forward_hook(get_input_hook(sample_inputs_dict, child_name))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            module.forward(*sample_inputs)
        for child_name, child_module in module.named_children():
            if isinstance(child_module, OperatorBase):
                print(f'########## {type(child_module)} {child_name} ##########')
                tune_sparse_module(child_module, sample_inputs_dict[child_name])


def tune_sparse_module(operator: OperatorBase, sample_inputs: List[torch.Tensor]):
    '''Tune and build the given sparse operator.

    Args:
        module (OperatorBase): A tunable sparse operator.
        sample_inputs (List[torch.Tensor]): Sample input tensors to determine shape parameters.
    '''
    best_params = operator.tune(sample_inputs)
    if best_params is None:
        warnings.warn('All trails failed, please re-tune with a different search space.')
    else:
        operator.build(best_params, sample_inputs=sample_inputs)


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
