# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import abc
import json
import copy
import shutil
import hashlib
import subprocess
import collections

import torch
import numpy as np
from jinja2 import Template

# from SparTA.Specializer import TeSA
import TeSA


CUDA_TEMPLATE_DIR = os.path.join('SparTA', 'Specializer', 'Templates')
OPERATOR_CONFIG_DIR = os.path.join('SparTA', 'Specializer', 'Operators')


class Operator(object):

    def __init__(self, op_config_file='SparseLinear'):
        with open(os.path.join(OPERATOR_CONFIG_DIR, f'{op_config_file}.json')) as f:
            op_config = json.loads(f.read())
        self.name = op_config['name']
        self.dims = op_config['dims']
        self.inputs = op_config['inputs']
        self.outputs = op_config['outputs']
        self.tiles = op_config['tiles']
        self.gpu_code = '61'
        with open(os.path.join(CUDA_TEMPLATE_DIR, 'Kernels', f'{self.name}.cuh.j2')) as f:
            self.template = f.read()


class KernelInterface(abc.ABC):

    def __init__(self, operator: 'Operator', shape_config: dict):
        self._operator = operator
        self._id = hashlib.sha1(str(sorted(shape_config.items())).encode()).hexdigest()[:6]
        self._dir = os.path.join('tmp', operator.name, self._id)
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        self._shape = copy.deepcopy(shape_config)
        self._data = {}
        self._config = self._load_codes()
        self._config |= self._load_data_config()
        self._config |= self._load_tile_config()
        self._build()

    def _load_codes(self) -> dict[str, str]:
        function_body = Template(self._operator.template).render(self._shape)
        function_name = function_body[function_body.find('__global__ void') + 15:]
        function_name = function_name[:function_name.find('(')].strip()
        return {
            'KERNEL_FUNC_NAME': function_name,
            'KERNEL_FUNC_BODY': function_body
        }

    def _expand_data_config(self, raw_data_config: dict) -> dict:
        data_config = copy.deepcopy(raw_data_config)
        for key, settings in data_config.items():
            if 'name' not in settings.keys():
                settings['name'] = key
            if 'filepath' not in settings.keys():
                settings['filepath'] = os.path.join(self._dir, f'{settings["name"]}.dat')
        return data_config

    def _expand_tesa_config(self, raw_data_config) -> dict:
        data_config = {}
        for data_k, data_v in raw_data_config.items():
            if data_v['layout'] == 'bcsr':
                for tesa_k, tesa_v in TeSA.BCSR.desc().items():
                    desc = copy.deepcopy(data_v)
                    desc.update(tesa_v)
                    data_config[f'{data_k}_{tesa_k}'] = desc
            else:
                data_config[data_k] = data_v
        return self._expand_data_config(data_config)

    def _load_data_config(self) -> dict:
        self._inputs = self._expand_data_config(self._operator.inputs)
        self._outputs = self._expand_data_config(self._operator.outputs)
        return {
            'INPUTS': self._expand_tesa_config(self._operator.inputs).values(),
            'OUTPUTS': self._expand_tesa_config(self._operator.outputs).values(),
        }

    def _replace_and_eval(self, s: str) -> int:
        for k, v in self._shape.items():
            s = s.replace(k, str(v))
        for k in self._data.keys():
            s = s.replace(k, f'self._data["{k}"]')
        return eval(s)

    def _load_tile_config(self) -> dict[str, list[int]]:
        return {
            'DIM_BLOCK': list(map(self._replace_and_eval, self._operator.tiles['block'])),
            'DIM_GRID': list(map(self._replace_and_eval, self._operator.tiles['grid']))
        }

    def _run_cmd(self, cmd: str) -> str:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout = stdout.decode("utf-8").replace('\\n', '\n')
        stderr = stderr.decode("utf-8").replace('\\n', '\n')
        if len(stderr) > 0:
            raise subprocess.SubprocessError(stderr)
        return stdout

    @abc.abstractclassmethod
    def _build(self):
        '''
        Build kernel function
        '''

    @abc.abstractclassmethod
    def __call__(self, *args):
        '''
        Function call
        '''


class TestInterface(KernelInterface):

    def _build(self):
        with open(os.path.join(CUDA_TEMPLATE_DIR, 'test.cu.j2')) as f:
            test_template = f.read()
        test_code = Template(test_template).render(self._config)
        self._code_path = os.path.join(self._dir, f'{self._operator.name}.cu')
        self._exec_path = os.path.join(self._dir, self._operator.name)
        with open(self._code_path, 'w') as f:
            f.write(test_code)
        gpu_code = self._operator.gpu_code
        self._run_cmd(f"nvcc -gencode arch=compute_{gpu_code},code=sm_{gpu_code} {self._code_path} -w -o {self._exec_path}")

    def _generate_data(self, desc: dict) -> 'np.ndarray':
        if 'formula' in desc.keys():
            val = self._replace_and_eval(desc['formula'])
            if val.shape != tuple(map(self._replace_and_eval, desc['shape'])):
                raise ValueError(f'{desc.keys()} formula and shape do not match')
        else:
            val = np.random.normal(size=list(map(self._replace_and_eval, desc['shape'])))
        return val.astype(f'{desc["type"]}32')

    def _import_data(self, desc: dict, val: 'np.ndarray'):
        self._data[desc["name"]] = val
        if desc['layout'] == 'dense':
            self._save_data(desc["name"], val)
        elif desc['layout'] == 'bcsr':
            height, width = val.shape  # TODO: support high-dim tensors
            block_height, block_width = self._config['DIM_BLOCK']
            row_num, col_num = height // block_height, width // block_width
            mask = np.random.uniform(size=(row_num, col_num)) < 0.2
            bcsr = TeSA.BCSR(val, mask=mask, block_width=block_width, block_height=block_height)
            for k, v in bcsr.tesa().items():
                self._save_data(f'{desc["name"]}_{k}', v)

    def _save_data(self, name: str, val: 'np.ndarray'):
        with open(os.path.join(self._dir, f'{name}.dat'), 'wb') as f:
            val.flatten().tofile(f)

    def __call__(self, *args) -> float:
        self._data = {}
        for i, desc in enumerate((self._inputs | self._outputs).values()):
            self._import_data(desc, args[i] if i < len(args) else self._generate_data(desc))
        result = self._run_cmd(self._exec_path)
        try:
            return float(result)
        except ValueError:
            return float("inf")

    # def __del__(self):
    #     shutil.rmtree(self._dir, ignore_errors=True)


class ModuleInterface(KernelInterface, torch.nn.Module):

    def _build(self):
        pass

    def __call__(self, *args):
        pass


M, K, N = 1024, 256, 512
BM, BK, BN = 64, 8, 128
TM, TK, TN = 8, 4, 8
op = Operator(op_config_file='SparseLinear')
f = TestInterface(op, {
    'GLOBAL_M_VALUE': M,
    'GLOBAL_K_VALUE': K,
    'GLOBAL_N_VALUE': N,
    'BLOCK_SIZE_M_VALUE': BM,
    'BLOCK_SIZE_K_VALUE': BK,
    'BLOCK_SIZE_N_VALUE': BN,
    'THREAD_SIZE_M_VALUE': TM,
    'THREAD_SIZE_K_VALUE': TK,
    'THREAD_SIZE_N_VALUE': TN
})
print(f())