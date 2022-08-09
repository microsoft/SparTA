# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import abc
import copy
import shutil
import hashlib
import subprocess
from typing import Any, List, Dict, Callable

import torch
import jinja2
import numpy as np
from torch.utils import cpp_extension

from sparta.common import tesa, utils


COMMON_TEMPLATE_DIR = os.path.join('sparta', 'specializer', 'factories', 'templates')


def get_tesa_class(layout: str):
    if layout == 'bcsr':
        return tesa.BCSR
    else:
        raise ValueError(f'Unrecognized TeSA layout: {layout}')


class FactoryBase(abc.ABC):

    def __init__(self, op_config: Dict):
        self.op_name = op_config['op_name']
        self.kernel_name = op_config['kernel_name']
        self.dynamic_dims = op_config['dynamic_dims']
        self.fixed_dims = op_config['fixed_dims']
        self.inputs = op_config['inputs']
        self.outputs = op_config['outputs']
        self.tiles = op_config['tiles']

    @abc.abstractclassmethod
    def get_kernel_code(self, **kwargs) -> str:
        '''
        Get CUDA code of the kernel
        '''

    def get_test_func(self, shape_config: Dict, mask: Dict[str, np.ndarray] = None) -> 'TestInterface':
        return TestInterface(self, shape_config, mask)

    def get_module(self, shape_config: Dict, mask: Dict[str, np.ndarray] = None) -> 'ModuleInterface':
        return ModuleInterface(self, shape_config, mask).get_module()

    def get_module_code(self, shape_config: Dict, mask: Dict[str, np.ndarray] = None) -> str:
        return ModuleInterface(self, shape_config, mask).get_module_code()


class KernelInterface(abc.ABC):

    def __init__(self, factory: FactoryBase, shape_config: Dict, mask: Dict[str, np.ndarray]):
        self._factory = factory
        self._id = hashlib.sha1(str(sorted(shape_config.items())).encode()).hexdigest()[:6]
        self._shape = copy.deepcopy(shape_config)
        self._mask = {} if mask is None else mask
        self._data = {}
        self._config = self._load_codes()
        self._config |= self._load_data_desc()
        self._build()

    def _load_codes(self) -> Dict[str, str]:
        function_body = self._factory.get_kernel_code(**(self._shape | self._factory.fixed_dims))
        function_name = function_body[function_body.find('__global__ void') + 15:]
        function_name = function_name[:function_name.find('(')].strip()
        return {
            'KERNEL_FUNC_NAME': function_name,
            'KERNEL_FUNC_BODY': function_body
        }

    def _expand_data_desc(self, raw_desc_dict: Dict) -> Dict:
        desc_dict = copy.deepcopy(raw_desc_dict)
        for name, desc in desc_dict.items():
            desc['name'] = name
            desc['shape'] = list(map(self._replace_and_eval, desc['shape']))
        return desc_dict

    def _expand_tesa_desc(self, raw_desc_dict: Dict) -> Dict:
        desc_dict = {}
        for data_name, data_desc in raw_desc_dict.items():
            data_desc['role'] = 'data'
            if data_desc['layout'] == 'dense':
                desc_dict[data_name] = data_desc
            else:
                TeSAClass = get_tesa_class(data_desc['layout'])
                mode = data_desc['mode'] if 'mode' in data_desc else 'H'
                for tesa_name, tesa_desc in TeSAClass.desc(mode=mode).items():
                    desc = copy.deepcopy(data_desc)
                    desc.update(tesa_desc)
                    desc_dict[f'{data_name}_{tesa_name}'] = desc
        return self._expand_data_desc(desc_dict)

    def _load_data_desc(self) -> Dict:
        self._inputs = self._expand_data_desc(self._factory.inputs)
        self._outputs = self._expand_data_desc(self._factory.outputs)
        input_descs = []
        output_descs = []
        for desc in self._expand_tesa_desc(self._factory.inputs).values():
            input_descs.append(desc)
        for desc in self._expand_tesa_desc(self._factory.outputs).values():
            if desc['role'] == 'tesa':
                input_descs.append(desc)
            else:
                output_descs.append(desc)
        return {'INPUTS': input_descs, 'OUTPUTS': output_descs}

    def _replace_and_eval(self, s: str) -> Any:
        def find_data(x: str):
            for k, v in self._shape.items():
                if k == x:
                    return str(v)
            for k in self._data:
                if k == x:
                    return f'self._data["{k}"]'
            return x
        t = ''
        x = ''
        for c in str(s):
            if c.isalpha() or c.isnumeric() or c == '_':
                x += c
            else:
                t += find_data(x)
                t += c
                x = ''
        t += find_data(x)
        return eval(t)

    def _load_tile_config(self) -> Dict[str, List[int]]:
        return {
            'DIM_BLOCK': list(map(self._replace_and_eval, self._factory.tiles['block'])),
            'DIM_GRID': list(map(self._replace_and_eval, self._factory.tiles['grid']))
        }

    def _convert_dense_data(self, desc: Dict, dense_val: np.ndarray) -> Dict[str, np.ndarray]:
        if desc['layout'] == 'dense':
            return {desc["name"]: dense_val}
        else:
            TeSAClass = get_tesa_class(desc['layout'])
            if desc['layout'].startswith('bcsr'):
                bcsr_cfg = {
                    'dense': dense_val,
                    'size': dense_val.shape,
                    'block_size': tuple(map(self._replace_and_eval, desc['block_size'])),
                    'mask': self._mask[desc['name']] if desc['name'] in self._mask else 0.2,
                    'mode': desc['mode'] if 'mode' in desc else 'H',
                }
                bcsr = TeSAClass(**bcsr_cfg)
                return {f'{desc["name"]}_{k}': v for k, v in bcsr.sparse.items()}

    def _run_cmd(self, cmd: str, timeout: float) -> str:
        process = subprocess.Popen(f'exec {cmd}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            process.kill()
            raise e
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


class TestInterface(KernelInterface, Callable):

    def _build(self):
        self._dir = os.path.join('tmp', self._factory.op_name, self._id)
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        self._specify_data_path(self._config['INPUTS'])
        self._specify_data_path(self._config['OUTPUTS'])
        with open(os.path.join(COMMON_TEMPLATE_DIR, 'test.cu.j2')) as f:
            self._template = f.read()
        self._code_path = os.path.join(self._dir, f'{self._factory.op_name}.cu')
        self._exec_path = os.path.join(self._dir, self._factory.op_name)

    def _build_exe(self):
        with open(self._code_path, 'w') as f:
            f.write(jinja2.Template(self._template).render(self._config | self._load_tile_config()))
        gpu_code = utils.cuda_detect()[0][1]
        build_args = f'arch=compute_{gpu_code},code=sm_{gpu_code}'
        self._run_cmd(
            f"nvcc -gencode {build_args} {self._code_path} -w -o {self._exec_path}",
            timeout=5
        )

    def _specify_data_path(self, desc_list: Dict) -> Dict:
        for desc in desc_list:
            if 'filepath' not in desc:
                desc['filepath'] = os.path.join(self._dir, f'{desc["name"]}.dat')

    def _generate_data(self, desc: Dict) -> np.ndarray:
        if 'formula' in desc:
            val = self._replace_and_eval(desc['formula'])
            if val.shape != tuple(desc['shape']):
                raise ValueError(
                    f'{desc["name"]} formula and shape do not match')
        else:
            val = np.random.normal(size=desc['shape'])
        return val.astype(f'{desc["type"]}32')

    def _import_data(self, desc: Dict, val: np.ndarray):
        self._data[desc["name"]] = val
        for k, v in self._convert_dense_data(desc, val).items():
            self._save_data(k, v)
            self._data[k] = v if v.size > 1 else v[0]  # TODO

    def _save_data(self, name: str, val: np.ndarray):
        with open(os.path.join(self._dir, f'{name}.dat'), 'wb') as f:
            val.flatten().tofile(f)

    def __call__(
        self, inputs: Dict[str, np.ndarray] = None, target_outputs: Dict[str, np.ndarray] = None,
        num_warmups: int = 10, num_iters: int = 10, check_results: bool = True
    ) -> float:
        raw_data = {}
        if inputs is not None:
            raw_data |= inputs
        if target_outputs is not None:
            raw_data |= target_outputs
        self._data = {}
        for name, desc in (self._inputs | (self._outputs if check_results else {})).items():
            data = raw_data[name] if name in raw_data else self._generate_data(desc)
            self._import_data(desc, data)
        self._build_exe()
        result = self._run_cmd(
            f'{self._exec_path} {num_warmups} {num_iters} {int(check_results)}',
            timeout=1
        )
        return float(result)

    # def __del__(self):
    #     shutil.rmtree(self._dir, ignore_errors=True)


class ModuleInterface(KernelInterface):

    def _build(self):
        mask_id = hashlib.sha1(str(self._mask).encode()).hexdigest()[:6]
        self._module_name = f'{self._factory.op_name}_{self._id}_{mask_id}'
        self._config['MODULE_NAME'] = self._module_name
        for input_desc in self._inputs.values():
            self._get_tesa_data(input_desc, 'INPUTS')
        for output_desc in self._outputs.values():
            self._get_tesa_data(output_desc, 'OUTPUTS')
        # print(self._config)

    def _get_tesa_data(self, desc: Dict[str, dict], category: str):
        fake_val = np.zeros(shape=desc['shape'])
        for k, v in self._convert_dense_data(desc, fake_val).items():
            for data_cfg in self._config['INPUTS'] + self._config['OUTPUTS']:
                if data_cfg['name'] == k:
                    data_cfg['shape'] = list(v.shape)
                    if data_cfg['role'] == 'tesa':
                        data_cfg['val'] = v.flatten().tolist()
                        self._data[k] = v if v.size > 1 else v[0]  # TODO
                    break

    def get_module_code(self):
        with open(os.path.join(COMMON_TEMPLATE_DIR, 'module.cu.j2')) as f:
            module_template = f.read()
        return jinja2.Template(module_template).render(self._config | self._load_tile_config())

    def get_module(self):
        return cpp_extension.load_inline(
            self._module_name,
            '',
            cuda_sources=self.get_module_code(),
            extra_cflags=['-O3'],
        )
