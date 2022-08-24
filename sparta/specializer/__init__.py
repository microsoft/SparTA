# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import subprocess
from typing import Dict, List, Optional

import numpy as np

from sparta.specializer import factories, tuners

KERNEL_CONFIG_DIR = os.path.join('sparta', 'specializer', 'configs', 'kernels')
OPERATOR_CONFIG_DIR = os.path.join('sparta', 'specializer', 'configs', 'operators')


def get_factory(kernel_name: str) -> factories.FactoryBase:
    kernel_config_file = f'{kernel_name}.json'
    with open(os.path.join(KERNEL_CONFIG_DIR, kernel_config_file)) as f:
        kernel_config = json.loads(f.read())
    factory_type = kernel_config['factory_type']
    if factory_type == 'template':
        return factories.TemplateBasedFactory(kernel_config)
    else:
        raise ValueError(f'Factory type "{factory_type}" is not supported.')


class Specializer(object):

    def __init__(
        self, op_name: str, shape: Optional[Dict[str, int]] = None,
        mask: Optional[Dict[str, np.ndarray]] = None
    ):
        op_config_file = f'{op_name}.json'
        with open(os.path.join(OPERATOR_CONFIG_DIR, op_config_file)) as f:
            op_config = json.loads(f.read())
        self._shape = op_config['default_shape']
        if shape is not None:
            self._shape.update(shape)
        self._mask = mask
        self._search_space: Dict[str, List[int]] = op_config['search_space']
        self._constraints: List[str] = op_config['constraints']
        self._special_kernels: Dict[str, List[str]] = op_config['special_kernels']
        self._default_kernel: str = op_config['default_kernel']

    def _check_config(self, config: Dict[str, int]):
        for constraint in self._constraints:
            for k, v in (self._shape | config).items():
                constraint = constraint.replace(k, str(v))
            if not eval(constraint):
                return False
        return True

    def _get_factory(self, config: Dict[str, int]):
        for kernel_name, conditions in self._special_kernels.items():
            flag = True
            for condition in conditions:
                for k, v in (self._shape | config).items():
                    condition = condition.replace(k, str(v))
                if not eval(condition):
                    flag = False
                    break
            if flag:
                return get_factory(kernel_name)
        return get_factory(self._default_kernel)

    def get_test_interface(self, config: Dict[str, int]):
        return self._get_factory(config).get_test_interface((self._shape | config), self._mask)

    def get_module_interface(self, config: Dict[str, int]):
        return self._get_factory(config).get_module_interface((self._shape | config), self._mask)

    def find_best_config(
            self, inputs: Optional[Dict[str, np.ndarray]] = None,
            tuner: str = 'grid', search_space: Optional[Dict[str, List[int]]] = None
        ) -> Optional[Dict[str, int]]:
        search_space = self._search_space if search_space is None else search_space
        if tuner == 'grid':
            tuner = tuners.GridSearchTunner(search_space=search_space)
        else:
            raise ValueError(f'unsupported tuner: {tuner}')
        best_cfg = None
        best_latency = float('inf')
        num = 0
        for cfg in tuner._configs():
            if self._check_config(cfg):
                num += 1
                print(f'#{num}: {", ".join([f"{k}={v}" for k, v in cfg.items()])}')
                try:
                    latency = self.get_test_interface(cfg)(inputs=inputs)
                except subprocess.SubprocessError:
                    print(f'An error occured')
                    continue
                if latency < best_latency:
                    best_cfg = cfg
                    best_latency = latency
                print(f'Latency: {latency} ms')
        if best_cfg is not None:
            print(f'Best config: {", ".join([f"{k}={v}" for k, v in best_cfg.items()])}')
        else:
            print('All configs test failed')
        return best_cfg
