# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import subprocess
from typing import Dict, List

from sparta.specializer import tuners

def print(s: str):
    with open('tuning_log.txt', 'a') as f:
        f.write(s + '\n')

class GridSearchTunner(tuners.TunerBase):

    def _generate_all_cfgs(self, keys: List[str], cfg: Dict[str, int] = {}):
        if len(keys) == 0:
            # TODO: Move to OP tuner config
            if (
                cfg['GLOBAL_M_VALUE'] > cfg['BLOCK_SIZE_M_VALUE'] and cfg['BLOCK_SIZE_M_VALUE'] > cfg['THREAD_SIZE_M_VALUE'] and
                cfg['GLOBAL_N_VALUE'] > cfg['BLOCK_SIZE_N_VALUE'] and cfg['BLOCK_SIZE_N_VALUE'] > cfg['THREAD_SIZE_N_VALUE'] and
                cfg['GLOBAL_K_VALUE'] > cfg['BLOCK_SIZE_K_VALUE'] and cfg['BLOCK_SIZE_K_VALUE'] > cfg['THREAD_SIZE_K_VALUE']
            ):
                return [copy.deepcopy(cfg)]
            else:
                return []
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
            try:
                latency = self._factory.get_test_func(_cfg)()
            except subprocess.SubprocessError:
                print(f'An error occured')
                continue
            if latency < best_latency:
                best_cfg = _cfg
                best_latency = latency
            print(f'Latency: {latency} ms')
        print(f'Best config: {best_cfg}')
        return best_cfg
