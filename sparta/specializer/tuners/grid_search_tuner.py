# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from abc import abstractmethod
from typing import Dict, List

from sparta.specializer import tuners


class GridSearchTunner(tuners.TunerBase):

    def _generate_all_cfgs(self, keys: List[str], cfg: Dict[str, int] = {}):
        if len(keys) == 0:
            return [copy.deepcopy(cfg)]
        key = keys.pop()
        cfgs = []
        for cfg[key] in self._search_space[key]:
            cfgs += self._generate_all_cfgs(copy.deepcopy(keys), cfg)
        return cfgs

    def tunning_kernel_cfg(self):
        cfg_space = self._generate_all_cfgs(list(self._search_space.keys()))
        print(f'Searching through {len(cfg_space)} configs...')
        best_cfg = None
        best_latency = float('inf')
        for i, _cfg in enumerate(cfg_space):
            print(f'{i + 1}/{len(cfg_space)}: {list(_cfg.values())}')
            latency = self._factory.get_test_func(_cfg)()
            if latency < best_latency:
                best_cfg = _cfg
                best_latency = latency
            print(f'Latency: {latency} s')
        print(f'Best config: {best_cfg}')
        return best_cfg
