# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
from typing import Dict, List, Type, Optional

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

    def __init__(self, op_name: str, *args, **kwargs):
        op_config_file = f'{op_name}.json'
        with open(os.path.join(OPERATOR_CONFIG_DIR, op_config_file)) as f:
            op_config = json.loads(f.read())
        hyper_params: List[str] = op_config['hyper_params'] if 'hyper_params' in op_config else []
        if len(hyper_params) > 0:
            if len(args) > 0:
                if len(args) != len(hyper_params):
                    raise TypeError(f'{op_name} specializer requires arguments: {hyper_params}')
                self._hyper_params = {hyper_params[i]: v for i, v in enumerate(args)}
            elif len(kwargs) > 0:
                if set(kwargs) != set(hyper_params):
                    raise TypeError(f'{op_name} specializer requires arguments: {hyper_params}')
                self._hyper_params = kwargs
            else:
                raise TypeError(f'{op_name} specializer requires arguments: {hyper_params}')
        self.search_space: Dict[str, List[int]] = op_config['search_space']
        self._constraints: List[str] = op_config['constraints']
        self._special_kernels: Dict[str, List[str]] = op_config['special_kernels']
        self._default_kernel: str = op_config['default_kernel']

    def _check_config(self, config: Dict[str, int]):
        for constraint in self._constraints:
            for k, v in (self._hyper_params | config).items():
                constraint = constraint.replace(k, str(v))
            if not eval(constraint):
                return False
        return True

    def _get_factory(self, config: Dict[str, int]):
        for kernel_name, conditions in self._special_kernels.items():
            flag = True
            for condition in conditions:
                for k, v in (self._hyper_params | config).items():
                    condition = condition.replace(k, str(v))
                if not eval(condition):
                    flag = False
                    break
            if flag:
                return get_factory(kernel_name)
        return get_factory(self._default_kernel)

    def get_test_func(self, config: Dict[str, int], mask: Optional[Dict[str, np.ndarray]] = None):
        return self._get_factory(config).get_test_func((self._hyper_params | config), mask)

    def get_module(self, config: Dict[str, int], mask: Optional[Dict[str, np.ndarray]] = None):
        return self._get_factory(config).get_module((self._hyper_params | config), mask)

    def get_module_code(self, config: Dict[str, int], mask: Optional[Dict[str, np.ndarray]] = None):
        return self._get_factory(config).get_module_code((self._hyper_params | config), mask)

    def find_best_config(
            self, inputs: Optional[Dict[str, np.ndarray]] = None,
            mask: Optional[Dict[str, np.ndarray]] = None,
            tuner_type: Type[tuners.TunerBase] = tuners.GridSearchTunner,
            search_space: Optional[Dict[str, List[int]]] = None
        ) -> Optional[Dict[str, int]]:
        search_space = self.search_space if search_space is None else search_space
        tuner = tuner_type(specializer=self, search_space=search_space)
        return tuner.find_best_config(inputs=inputs, mask=mask)
