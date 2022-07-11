# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import abc
import hashlib
import subprocess

import numpy as np
from jinja2 import Template

# from SparTA.Common.Utils import cuda_detect, call_shell


TEMPLATE_PATH = os.path.join('SparTA', 'Specializer', 'Templates')

class FactoryBase(abc.ABC):

    def __init__(self):
        self._set_attrs()
        # self._gpu_devices = cuda_detect()
        # assert len(self._gpu_devices) > 0
        # self._gpu_code = self._gpu_devices[0][1]
        self._gpu_code = '61'
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

    def _load_tile_config(self, config) -> dict[str, list[int]]:
        def replace_and_eval(s: str) -> int:
            for k, v in config.items():
                s = s.replace(k, str(v))
            return eval(s)
        return {
            'DIM_BLOCK': list(map(replace_and_eval, self._tiles['block'])),
            'DIM_GRID': list(map(replace_and_eval, self._tiles['grid']))
        }

    def get_test_function(self, config: dict[str, any]) -> 'TestFunction':
        unique_id = hashlib.sha1(str(sorted(config.items())).encode()).hexdigest()[:6]
        test_folder = os.path.join('tmp', self._operator_name, unique_id)
        for input_name, input_settings in self._inputs.items():
            if 'name' not in input_settings.keys():
                input_settings['name'] = input_name
            if 'filepath' not in input_settings.keys():
                input_settings['filepath'] = os.path.join(test_folder, f'{input_settings["name"]}.dat')
        for output_name, output_settings in self._outputs.items():
            if 'name' not in output_settings.keys():
                output_settings['name'] = output_name
            if 'filepath' not in output_settings.keys():
                output_settings['filepath'] = os.path.join(test_folder, f'{output_settings["name"]}.dat')
        kernel_function_body = Template(self._kernel_template).render(config)
        kernel_function_name = kernel_function_body[kernel_function_body.find('__global__ void') + 15:]
        kernel_function_name = kernel_function_name[:kernel_function_name.find('(')].strip()
        test_config = config | self._load_tile_config(config)
        test_config |= {
            'KERNEL_FUNC_NAME': kernel_function_name,
            'KERNEL_FUNC_BODY': kernel_function_body
        }
        return TestFunction(
            folder=test_folder,
            inputs=self._inputs,
            outputs=self._outputs,
            gpu_code=self._gpu_code,
            config=test_config
        )

    @abc.abstractmethod
    def test(self, *args, **kwargs):
        '''
        Do something with the test function
        '''


class TestFunction(object):

    def __init__(self, folder: str, inputs: dict, outputs: dict, config: dict, gpu_code='61'):
        with open(os.path.join(TEMPLATE_PATH, 'test_function.cu.j2')) as f:
            test_function_template = f.read()
        config |= {
            'INPUTS': inputs.values(),
            'OUTPUTS': outputs.values(),
        }
        test_function_code = Template(test_function_template).render(config)
        self._folder = folder
        self._inputs = inputs
        self._outputs = outputs
        if not os.path.exists(folder):
            os.makedirs(folder)
        self._code_path = os.path.join(folder, 'test_func.cu')
        self._exec_path = os.path.join(folder, 'test_func')
        with open(self._code_path, 'w') as f:
            f.write(test_function_code)
        self._run_cmd(f"nvcc -gencode arch=compute_{gpu_code},code=sm_{gpu_code} {self._code_path} -o {self._exec_path}")

    def save_test_data(self, input_data: dict[str, 'np.array'], output_data: dict[str, 'np.array']):
        for k, v in input_data.items():
            with open(os.path.join(self._folder, f'{self._inputs[k]["name"]}.dat'), 'wb') as f:
                v.flatten().astype(f'{self._inputs[k]["type"]}32').tofile(f)
        for k, v in output_data.items():
            with open(os.path.join(self._folder, f'{self._outputs[k]["name"]}.dat'), 'wb') as f:
                v.flatten().astype(f'{self._outputs[k]["type"]}32').tofile(f)

    def _run_cmd(self, cmd) -> str:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout = stdout.decode("utf-8").replace('\\n', '\n')
        stderr = stderr.decode("utf-8").replace('\\n', '\n')
        if process.returncode and len(stderr):
            raise subprocess.SubprocessError(stderr)
        return stdout

    def __call__(self) -> float:
        result = self._run_cmd(self._exec_path)
        try:
            return float(result)
        except ValueError:
            return float("inf")

    def __del__(self):
        pass
