# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import tempfile
from cmath import inf
import torch
import types
import logging

from .sparse_opbase import SparseOPBase
import sparse_moe
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class DynamicSparseMoE(SparseOPBase):
    def __init__(self, n_exp, exp_modules):
        super().__init__()
        self.n_exp = n_exp
        self.in_hidden = exp_modules[0].weight.size(1)
        self.out_hidden = exp_modules[0].weight.size(0)
        self.device = exp_modules[0].weight.device
        assert(len(exp_modules)==n_exp)
        self.weight = torch.nn.Parameter(torch.rand(self.n_exp, self.in_hidden, self.out_hidden).to(self.device))
        with torch.no_grad():
            for eid in range(self.n_exp):
                self.weight.data[eid] = exp_modules[eid].weight.t().data
        self.sparse_index = torch.zeros(self.n_exp, 4096, dtype=torch.int32).to(self.device)
        self.expert_count = torch.zeros(self.n_exp, dtype=torch.int32).to(self.device)

    def forward(self, tokens, expids):
        sparse_moe.convert_index(expids, self.sparse_index, self.expert_count)
        GLOBAL_M = torch.max(self.expert_count)
        return sparse_moe.forward(tokens, self.weight, expids, self.sparse_index, self.expert_count, GLOBAL_M)