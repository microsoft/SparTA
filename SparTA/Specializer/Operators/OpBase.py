# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import abc
import hashlib
from typing import Dict, List

from jinja2 import Template


TEMPLATE_PATH = os.path.join('SparTA', 'Specializer', 'Templates')

class FactoryBase(abc.ABC):

    def __init__(self):
        self._set_attrs()
        with open(os.path.join(TEMPLATE_PATH, 'Operators', f'{self._operator_name}.cu.j2')) as f:
            self._operator_template = f.read()
        with open(os.path.join(TEMPLATE_PATH, 'Kernels', f'{self._kernel_name}.cuh.j2')) as f:
            self._kernel_template = f.read()

    @abc.abstractmethod
    def _set_attrs():
        '''
        Define attributes.
            self._operator_name: string, operator name.
            self._kernel_name: string, kernel name.
            self._shape_features: string[], shape variables.
            self._inputs: {string: {'type': string, 'shape': string[], 'sparsity': string[]}}
            self._outputs: {string: {'type': string, 'shape': string[], 'sparsity': string[]}}
            self._tiles: {'block' | 'grid': string[]}
        '''

    def _load_tile_config(self, **kwargs) -> Dict[str, List[int]]:
        def replace_and_eval(s: str) -> int:
            for k, v in kwargs.items():
                s = s.replace(k, str(v))
            return eval(s)
        return {
            'DIM_BLOCK': list(map(replace_and_eval, self._tiles['block'])),
            'DIM_GRID': list(map(replace_and_eval, self._tiles['grid']))
        }

    def get_test_function(self, **kwargs) -> 'TestFunction':
        unique_id = hashlib.sha1(str(sorted(kwargs.items())).encode()).hexdigest()[:6]
        test_folder = os.path.join('tmp', self._operator_name, unique_id)
        for input_name, input_settings in self._inputs.items():
            if 'name' not in input_settings.keys():
                input_settings['name'] = input_name
            if 'filepath' not in input_settings.keys():
                input_settings['filepath'] = os.path.join(test_folder, f'{input_settings["name"]}.bin')
        for output_name, output_settings in self._outputs.items():
            if 'name' not in output_settings.keys():
                output_settings['name'] = output_name
            if 'filepath' not in output_settings.keys():
                output_settings['filepath'] = os.path.join(test_folder, f'{output_settings["name"]}.bin')
        kernel_function_body = Template(self._kernel_template).render(kwargs)
        kernel_function_name = kernel_function_body[kernel_function_body.find('__global__ void') + 15:]
        kernel_function_name = kernel_function_name[:kernel_function_name.find('(')].strip()
        return TestFunction(test_folder, **(kwargs | {
            'INPUTS': self._inputs.values(),
            'OUTPUTS': self._outputs.values(),
        } | self._load_tile_config(**kwargs) | {
            'KERNEL_FUNC_NAME': kernel_function_name,
            'KERNEL_FUNC_BODY': kernel_function_body
        }))

    @abc.abstractmethod
    def test(self, *args, **kwargs):
        '''
        Do something with the test function
        '''


class TestFunction(object):

    def __init__(self, folder: str, **kwargs):
        with open(os.path.join(TEMPLATE_PATH, 'test_function.cu.j2')) as f:
            test_function_template = f.read()
        test_function_code = Template(test_function_template).render(kwargs)
        self._folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, 'test.cu'), 'w') as f:
            f.write(test_function_code)

    def __call__(self) -> float:
        pass

    def __del__(self):
        pass
