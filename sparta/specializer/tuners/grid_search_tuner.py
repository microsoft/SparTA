# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from typing import Dict, List

from sparta.specializer import tuners

class GridSearchTunner(tuners.TunerBase):

    def _generate_all_cfgs(self, keys: List[str], cfg: Dict[str, int]):
        if len(keys) == 0:
            return [copy.deepcopy(cfg)]
        key = keys.pop()
        cfgs: List[Dict[str, int]] = []
        for cfg[key] in self._search_space[key]:
            cfgs += self._generate_all_cfgs(copy.deepcopy(keys), cfg)
        return cfgs

    def _configs(self):
        cfg_space = self._generate_all_cfgs(list(self._search_space.keys()), {})
        for cfg in cfg_space:
            yield cfg
