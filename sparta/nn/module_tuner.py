# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings

import torch

from sparta.specializer import OperatorBase, SparseLinear, SparseSoftmax


def tune_combined_module(module: torch.nn.Module, sample_inputs: list[torch.Tensor]):
    '''Find, tune and build all sparse operators in the model.

    Args:
        module (torch.nn.Module): A PyTorch module that contains one or more sparse sub-modules.
        sample_inputs (list[torch.Tensor]): Sample input tensors to determine shape parameters.
    '''
    if isinstance(module, OperatorBase):
        tune_sparse_module(module, sample_inputs)
    else:  # TODO: Input hook
        for name, operator in module.named_children():
            if isinstance(operator, OperatorBase):
                print(f'########## {type(operator)} {name} ##########')
                tune_sparse_module(operator, sample_inputs)

def tune_sparse_module(operator: OperatorBase, sample_inputs: list[torch.Tensor]):
    '''Tune and build the given sparse operator.

    Args:
        module (OperatorBase): A tunable sparse operator.
        sample_inputs (list[torch.Tensor]): Sample input tensors to determine shape parameters.
    '''
    best_impl, best_config = operator.tune(sample_inputs)
    if best_impl is None or best_config is None:
        warnings.warn('All trails failed, please re-tune with a different search space.')
    else:
        operator.build(best_impl, best_config)
