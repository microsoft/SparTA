# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from sparta.specializer import OperatorBase, SparseLinear, SparseSoftmax


def tune(module: torch.nn.Module):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, OperatorBase):
            print(f'########## Tunable sub-module: {name} ##########')
            best_config = sub_module.tune()
            sub_module.build(best_config)
