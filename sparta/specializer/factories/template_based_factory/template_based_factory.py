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
import jinja2
import numpy as np
from torch.utils import cpp_extension

from sparta.common import tesa, utils
from sparta.specializer import factories


CUDA_TEMPLATE_DIR = os.path.join('sparta', 'specializer', 'factories', 'template_based_factory', 'templates')


class TemplateBasedFactory(factories.FactoryBase):

    def __init__(self, op_config: dict):
        super().__init__(op_config)
        with open(os.path.join(CUDA_TEMPLATE_DIR, 'kernels', f'{self.name}.cuh.j2')) as f:
            self.template = f.read()

    def get_test_func(self, shape_config: dict, mask: dict[str, 'np.ndarray'] = None) -> Callable:
        return TestInterface(self, shape_config, mask)

    def get_module(self, shape_config: dict, mask: dict[str, 'np.ndarray'] = None) -> 'torch.nn.Module':
        return ModuleInterface(self, shape_config, mask)()


class KernelInterface(abc.ABC):

    def __init__(self, factory: 'TemplateBasedFactory', shape_config: dict, mask: dict[str, 'np.ndarray']):
        self._factory = factory
        self._id = hashlib.sha1(str(sorted(shape_config.items())).encode()).hexdigest()[:6]
        self._dir = os.path.join('tmp', factory.name, self._id)
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        self._shape = copy.deepcopy(shape_config)
        self._mask = {} if mask is None else mask
        self._data = {}
        self._config = self._load_codes()
        self._config |= self._load_data_desc()
        self._config |= self._load_tile_config()
        self._build()

    def _load_codes(self) -> dict[str, str]:
        function_body = jinja2.Template(self._factory.template).render(self._shape)
        function_name = function_body[function_body.find('__global__ void') + 15:]
        function_name = function_name[:function_name.find('(')].strip()
        return {
            'KERNEL_FUNC_NAME': function_name,
            'KERNEL_FUNC_BODY': function_body
        }

    def _expand_data_desc(self, raw_desc_dict: dict) -> dict:
        desc_dict = copy.deepcopy(raw_desc_dict)
        for name, desc in desc_dict.items():
            desc['name'] = name
            desc['shape'] = list(map(self._replace_and_eval, desc['shape']))
            if 'filepath' not in desc:
                desc['filepath'] = os.path.join(self._dir, f'{desc["name"]}.dat')
        return desc_dict

    def _expand_tesa_desc(self, raw_desc_dict) -> dict:
        desc_dict = {}
        for data_name, data_desc in raw_desc_dict.items():
            data_desc['role'] = 'data'
            if data_desc['layout'] == 'bcsr':
                for tesa_name, tesa_desc in tesa.BCSR.desc().items():
                    desc = copy.deepcopy(data_desc)
                    desc.update(tesa_desc)
                    desc_dict[f'{data_name}_{tesa_name}'] = desc
            else:
                desc_dict[data_name] = data_desc
        return self._expand_data_desc(desc_dict)

    def _load_data_desc(self) -> dict:
        self._inputs = self._expand_data_desc(self._factory.inputs)
        self._outputs = self._expand_data_desc(self._factory.outputs)
        return {
            'INPUTS': list(self._expand_tesa_desc(self._factory.inputs).values()),
            'OUTPUTS': list(self._expand_tesa_desc(self._factory.outputs).values()),
        }

    def _replace_and_eval(self, s: str) -> int:
        for k, v in self._shape.items():
            s = s.replace(k, str(v))
        for k in self._data:
            s = s.replace(k, f'self._data["{k}"]')
        return eval(s)

    def _load_tile_config(self) -> dict[str, list[int]]:
        return {
            'DIM_BLOCK': list(map(self._replace_and_eval, self._factory.tiles['block'])),
            'DIM_GRID': list(map(self._replace_and_eval, self._factory.tiles['grid']))
        }

    def _convert_dense_data(self, desc: dict, val: 'np.ndarray') -> dict[str, 'np.ndarray']:
        if desc['layout'] == 'bcsr':
            height, width = desc['shape']
            block_height, block_width = tuple(map(self._replace_and_eval, desc['block_size']))
            row_num, col_num = height // block_height, width // block_width
            if desc['name'] in self._mask:
                mask = self._mask[desc['name']]
            else:
                mask = np.random.uniform(size=(row_num, col_num)) < 0.2
            bcsr = tesa.BCSR(val, mask=mask, block_width=block_width, block_height=block_height)
            return {f'{desc["name"]}_{k}': v for k, v in bcsr.tesa().items()}
        else:
            return {desc["name"]: val}

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
    def __call__(self, *args, **kwargs):
        '''
        Function call
        '''


class TestInterface(KernelInterface):

    def _build(self):
        with open(os.path.join(CUDA_TEMPLATE_DIR, 'test.cu.j2')) as f:
            test_template = f.read()
        test_code = jinja2.Template(test_template).render(self._config)
        self._code_path = os.path.join(self._dir, f'{self._factory.name}.cu')
        self._exec_path = os.path.join(self._dir, self._factory.name)
        with open(self._code_path, 'w') as f:
            f.write(test_code)
        gpu_code = utils.cuda_detect()[0][1]
        self._run_cmd(f"nvcc -gencode arch=compute_{gpu_code},code=sm_{gpu_code} {self._code_path} -w -o {self._exec_path}")

    def _generate_data(self, desc: dict) -> 'np.ndarray':
        if 'formula' in desc:
            val = self._replace_and_eval(desc['formula'])
            if val.shape != tuple(desc['shape']):
                raise ValueError(f'{desc["name"]} formula and shape do not match')
        else:
            val = np.random.normal(size=desc['shape'])
        return val.astype(f'{desc["type"]}32')

    def _import_data(self, desc: dict, val: 'np.ndarray'):
        self._data[desc["name"]] = val
        for k, v in self._convert_dense_data(desc, val).items():
            self._save_data(k, v)

    def _save_data(self, name: str, val: 'np.ndarray'):
        with open(os.path.join(self._dir, f'{name}.dat'), 'wb') as f:
            val.flatten().tofile(f)

    def __call__(self, *args) -> float:
        self._data = {}
        for i, desc in enumerate((self._inputs | self._outputs).values()):
            self._import_data(desc, args[i] if i < len(args) else self._generate_data(desc))
        try:
            result = self._run_cmd(self._exec_path)  # TODO add args
            return float(result)
        except subprocess.SubprocessError:
            return float("inf")

    # def __del__(self):
    #     shutil.rmtree(self._dir, ignore_errors=True)


class ModuleInterface(KernelInterface):

    def _build(self):  # TODO public access to code
        module_name = f'{self._factory.name}_{self._id}'
        self._config['MODULE_NAME'] = module_name
        for input_desc in self._inputs.values():
            self._get_tesa_data(input_desc, 'INPUTS')
        for output_desc in self._outputs.values():
            self._get_tesa_data(output_desc, 'OUTPUTS')
        with open(os.path.join(CUDA_TEMPLATE_DIR, 'module.cu.j2')) as f:
            module_template = f.read()
        module_code = jinja2.Template(module_template).render(self._config)
        with open(os.path.join(self._dir, f'{self._factory.name}_bind.cu'), 'w') as f:
            f.write(module_code)
        self._module = cpp_extension.load_inline(
            module_name,
            '',
            cuda_sources=module_code,
            extra_cflags=['-O3'],
        )

    def _get_tesa_data(self, desc: dict[str, dict], category: str):
        fake_val = np.zeros(shape=desc['shape'])
        for k, v in self._convert_dense_data(desc, fake_val).items():
            idx = list(map(lambda x: x['name'], self._config[category])).index(k)
            self._config[category][idx]['shape'] = list(v.shape)
            if self._config[category][idx]['role'] == 'tesa':
                self._config[category][idx]['val'] = v.flatten().tolist()

    def __call__(self, *args):
        return self._module
