# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import abc
import copy
import shutil
import hashlib
import subprocess
from typing import Callable

import torch
import numpy as np
from jinja2 import Template

from SparTA.Specializer import TeSA, Utils
from SparTA.Specializer.Factories.FactoryBase import FactoryBase


CUDA_TEMPLATE_DIR = os.path.join('SparTA', 'Specializer', 'Factories', 'TemplateBasedFactory', 'Templates')


class TemplateBasedFactory(FactoryBase):

    def __init__(self, op_config: dict):
        super().__init__(op_config)
        with open(os.path.join(CUDA_TEMPLATE_DIR, 'Kernels', f'{self.name}.cuh.j2')) as f:
            self.template = f.read()

    def get_test_func(self, shape_config: dict) -> Callable:
        return TestInterface(self, shape_config)

    def get_module(self, shape_config: dict) -> 'torch.nn.Module':
        return ModuleInterface(self, shape_config)


class KernelInterface(abc.ABC):

    def __init__(self, factory: 'TemplateBasedFactory', shape_config: dict):
        self._factory = factory
        self._id = hashlib.sha1(str(sorted(shape_config.items())).encode()).hexdigest()[:6]
        self._dir = os.path.join('tmp', factory.name, self._id)
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        self._shape = copy.deepcopy(shape_config)
        self._data = {}
        self._config = self._load_codes()
        self._config |= self._load_data_config()
        self._config |= self._load_tile_config()
        self._build()

    def _load_codes(self) -> dict[str, str]:
        function_body = Template(self._factory.template).render(self._shape)
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
        self._inputs = self._expand_data_config(self._factory.inputs)
        self._outputs = self._expand_data_config(self._factory.outputs)
        return {
            'INPUTS': self._expand_tesa_config(self._factory.inputs).values(),
            'OUTPUTS': self._expand_tesa_config(self._factory.outputs).values(),
        }

    def _replace_and_eval(self, s: str) -> int:
        for k, v in self._shape.items():
            s = s.replace(k, str(v))
        for k in self._data.keys():
            s = s.replace(k, f'self._data["{k}"]')
        return eval(s)

    def _load_tile_config(self) -> dict[str, list[int]]:
        return {
            'DIM_BLOCK': list(map(self._replace_and_eval, self._factory.tiles['block'])),
            'DIM_GRID': list(map(self._replace_and_eval, self._factory.tiles['grid']))
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
        self._code_path = os.path.join(self._dir, f'{self._factory.name}.cu')
        self._exec_path = os.path.join(self._dir, self._factory.name)
        with open(self._code_path, 'w') as f:
            f.write(test_code)
        gpu_code = Utils.cuda_detect()[0][1]
        self._run_cmd(f"nvcc -gencode arch=compute_{gpu_code},code=sm_{gpu_code} {self._code_path} -w -o {self._exec_path}")

    def _generate_data(self, desc: dict) -> 'np.ndarray':
        if 'formula' in desc.keys():
            val = self._replace_and_eval(desc['formula'])
            if val.shape != tuple(map(self._replace_and_eval, desc['shape'])):
                raise ValueError(f'{desc.keys()} formula and shape do not match')
        else:
            np.random.seed(2022)
            val = np.random.normal(size=list(map(self._replace_and_eval, desc['shape'])))
        return val.astype(f'{desc["type"]}32')

    def _import_data(self, desc: dict, val: 'np.ndarray'):
        self._data[desc["name"]] = val
        if desc['layout'] == 'dense':
            self._save_data(desc["name"], val)
        elif desc['layout'] == 'bcsr':
            height, width = val.shape
            block_height, block_width = tuple(map(self._replace_and_eval, desc['block_size']))
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

    def __del__(self):
        shutil.rmtree(self._dir, ignore_errors=True)


class ModuleInterface(KernelInterface, torch.nn.Module):

    def _build(self):
        pass

    def __call__(self, *args):
        pass
