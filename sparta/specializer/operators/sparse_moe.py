# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch

import sparse_moe_cpp


class DynamicSparseMoE(torch.nn.Module):

    def __init__(self, exp_modules: List[torch.nn.Linear]):
        super().__init__()
        self.num_exps = len(exp_modules)
        self.in_hidden = exp_modules[0].weight.size(1)
        self.out_hidden = exp_modules[0].weight.size(0)
        self.device = exp_modules[0].weight.device
        with torch.no_grad():
            self.weight = torch.nn.Parameter(
                torch.stack([exp.weight.T for exp in exp_modules]).contiguous()
            )
        self.sparse_index = torch.zeros(self.num_exps, 4096, dtype=torch.int32, device=self.device)
        self.expert_count = torch.zeros(self.num_exps, dtype=torch.int32, device=self.device)

    def forward(self, tokens: torch.Tensor, exp_ids: torch.Tensor):
        sparse_moe_cpp.convert_index(exp_ids, self.sparse_index, self.expert_count)
        return sparse_moe_cpp.forward(
            tokens,
            self.weight,
            exp_ids,
            self.sparse_index,
            self.expert_count,
            torch.max(self.expert_count),
        )
