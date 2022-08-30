# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import abc
import copy
import glob
import hashlib
import subprocess
from typing import Any, Union, Optional, Callable, Iterable
from dataclasses import dataclass

import torch
import jinja2
import numpy as np
from torch.utils import cpp_extension

from sparta.common import tesa, utils


TEMPLATE_DIR = os.path.join('sparta', 'specializer', 'kernels', 'templates')

@dataclass
class _Parameter:
    name: str
    value: Any
    is_tunable: bool
    search_space: Optional[list[Any]] = None

    def __post_init__(self):
        if self.search_space is not None:
            assert self.is_tunable


@dataclass
class _Tensor:
    name: str
    dtype: str
    layout: str
    shape: Optional[tuple[str]] = None
    mask: Optional[np.ndarray] = None
    layout_config: Optional[dict] = None
    dense_data: Optional[np.ndarray] = None
    sparse_data: Optional[dict[str, np.ndarray]] = None

    def set_data(self, data: np.ndarray):
        assert data.shape == self.shape
        self.dense_data = data
        self.sparse_data = None
        if self.mask is not None:
            self.dense_data *= self.mask

    def generate_data(self):
        assert self.shape is not None
        self.dense_data = np.random.uniform(size=self.shape).astype(f'{self.dtype}32')
        if self.layout != 'dense':
            if self.mask is None:
                self.generate_mask()

    def generate_mask(self):
        assert self.shape is not None
        assert self.layout_config is not None
        if self.layout == 'BCSR':
            block_size = self.layout_config['block_size']
            row_num = self.shape[-2] // block_size[0]
            col_num = self.shape[-1] // block_size[1]
            mask = np.random.uniform(size=(row_num, col_num)) < 0.2
            mask = np.tile(mask.reshape((row_num, col_num, 1, 1)), [1, 1] + block_size)
            self.mask = mask.swapaxes(1, 2).reshape(self.shape)
        else:
            raise ValueError(f'invalid layout: {self.layout}')
        if self.dense_data is not None:
            self.dense_data *= self.mask

    def dense(self):
        if self.dense_data is not None:
            if self.mask is not None:
                self.dense_data *= self.mask
            return self.dense_data
        assert self.shape is not None
        assert self.layout_config is not None
        if self.layout == 'BCSR':
            self.dense_data = tesa.BCSR(
                size = self.shape,
                mask = self.mask,
                **(self.sparse_data | self.layout_config)
            ).dense
            return self.dense_data
        else:
            raise ValueError(f'invalid layout: {self.layout}')

    def sparse(self):
        if self.sparse_data is not None:
            return self.sparse_data
        assert self.dense_data is not None
        assert self.layout_config is not None
        if self.layout == 'BCSR':
            self.sparse_data = tesa.BCSR(
                dense = self.dense_data,
                size = self.shape,
                mask = self.mask,
                **(self.layout_config)
            ).sparse
            return self.sparse_data
        else:
            raise ValueError(f'invalid layout: {self.layout}')

    def sparse_desc(self):
        assert self.layout != 'dense'
        if self.layout == 'BCSR':
            return tesa.BCSR.desc(self.layout_config['mode'])
        else:
            raise ValueError(f'invalid layout: {self.layout}')


class KernelBase:

    def __init__(self):
        self.parameters: dict[str, _Parameter] = {}
        self.inputs: dict[str, _Tensor] = {}
        self.outputs: dict[str, _Tensor] = {}
        self.add_parameters()
        self.add_ports()

    def add_parameter(
        self, name: str, value: Any = None, is_tunable: bool = False,
        search_space: Optional[list[Any]] = None
    ):
        self.parameters[name] = _Parameter(name, value, is_tunable, search_space)

    def set_search_space(self, search_space: dict[str, list[Any]]):
        for name, space in search_space.items():
            self.parameters[name].search_space = space

    def get_search_space(self):
        return {p.name: p.search_space for p in self.parameters.values() if p.is_tunable}

    def set_parameter(self, name, value):
        self.parameters[name].value = value

    def get_parameter(self, name):
        return self.parameters[name].value

    def get_parameters(self):
        return {k: v.value for k, v in self.parameters.items()}

    def add_input(self, name: str, dtype: str, layout: str = 'dense'):
        self.inputs[name] = _Tensor(name, dtype, layout)

    def set_input_shape(self, name: str, shape: tuple[str]):
        self.inputs[name].shape = shape

    def set_input_layout(self, name: str, layout_config: dict):
        self.inputs[name].layout_config = layout_config

    def set_input(self, name: str, data: np.ndarray):
        self.inputs[name].set_data(data)

    def get_input(self, name: str):
        return self.inputs[name]

    def add_output(self, name: str, dtype: str, layout: str = 'dense'):
        self.outputs[name] = _Tensor(name, dtype, layout)

    def set_output_shape(self, name: str, shape: tuple[str]):
        self.outputs[name].shape = shape

    def set_output_layout(self, name: str, layout_config: dict):
        self.outputs[name].layout_config = layout_config

    def set_target_output(self, name: str, data: np.ndarray):
        self.outputs[name].set_data(data)

    def get_output(self, name: str):
        return self.outputs[name]

    def set_mask(self, mask: Optional[dict[str, np.ndarray]] = None, generate_if_missing = True):
        if mask is not None:
            for k, v in mask.items():
                if k in self.inputs:
                    self.inputs[k].mask = v
                elif k in self.outputs:
                    self.outputs[k].mask = v
        for input_tensor in self.inputs.values():
            if input_tensor.layout != 'dense' and input_tensor.mask is None:
                if generate_if_missing:
                    input_tensor.generate_mask()
                else:
                    raise ValueError(f'Missing mask on input tensor {input_tensor.name}')
        for output_tensor in self.outputs.values():
            if output_tensor.layout != 'dense' and output_tensor.mask is None:
                if generate_if_missing:
                    output_tensor.generate_mask()
                else:
                    raise ValueError(f'Missing mask on output tensor {output_tensor.name}')

    def configure(
        self, config: dict, mask: Optional[dict[str, np.ndarray]],
        generate_mask_if_missing: bool
    ) -> str:
        for k, v in config.items():
            self.set_parameter(k, v)
        self.set_ports_shape()
        self.set_ports_layout()
        self.set_mask(mask, generate_mask_if_missing)
        self.check_parameters()
        unique_id = self.get_kernel_name()
        unique_id += '_' + hashlib.sha1(str(sorted(config.items())).encode()).hexdigest()[:6]
        if not generate_mask_if_missing:
            assert mask is not None
            mask_str = ','.join([str(t.mask.tolist()) for t in self.inputs.values() if t.mask is not None])
            mask_str += ','.join([str(t.mask.tolist()) for t in self.outputs.values() if t.mask is not None])
            unique_id += hashlib.sha1(mask_str.encode()).hexdigest()[:6]
        return unique_id

    def test(
        self, config: dict, mask: Optional[dict[str, np.ndarray]] = None,
        inputs: dict[str, np.ndarray] = None, target_outputs: dict[str, np.ndarray] = None,
        num_warmups: int = 10, num_iters: int = 10, check_results: bool = True
    ) -> float:
        unique_id = self.configure(config, mask, True)
        if inputs is not None:
            for k, v in inputs.items():
                self.set_input(k, v)
        for input_tensor in self.inputs.values():
            if input_tensor.dense_data is None and input_tensor.sparse_data is None:
                input_tensor.generate_data()
        test_inputs = self.inputs.values()
        if target_outputs is not None:
            for k, v in target_outputs.items():
                self.set_target_output(k, v)
        elif check_results:
            self.calc_target_outputs()
        test_outputs = self.outputs.values() if check_results else None
        test_func = TestInterface(
            unique_id = unique_id,
            kernel_code = self.get_kernel_code(),
            shape = config,
            threads_per_block = self.threads_per_block(),
            blocks_per_grid = self.blocks_per_grid(),
            inputs = self.inputs.values(),
            outputs = self.outputs.values()
        )
        return test_func(test_inputs, test_outputs, num_warmups, num_iters, check_results)

    def compile(self, config: dict, mask: dict[str, np.ndarray], jit: bool = True):
        unique_id = self.configure(config, mask, False)
        for input_tensor in self.inputs.values():
            input_tensor.generate_data()
        for output_tensor in self.outputs.values():
            output_tensor.generate_data()
        return ModuleInterface(
            unique_id,
            self.get_kernel_code(),
            config,
            self.threads_per_block(),
            self.blocks_per_grid(),
            self.inputs.values(),
            self.outputs.values()
        ).get_module()

    @abc.abstractmethod
    def add_parameters(self):
        '''
        Add kernel-specialized parameters
        '''

    @abc.abstractmethod
    def check_parameters(self):
        '''
        Check if parameters are valid
        '''

    @abc.abstractmethod
    def add_ports(self):
        '''
        Add kernel-specialized inputs & outputs
        '''

    @abc.abstractmethod
    def set_ports_shape(self):
        '''
        Set shapes of inputs and outputs using determined parameters
        '''

    @abc.abstractmethod
    def set_ports_layout(self):
        '''
        Set layout configs of inputs and outputs using determined parameters
        '''

    @abc.abstractmethod
    def get_kernel_name(self) -> str:
        '''
        Get kernel name
        '''

    @abc.abstractmethod
    def get_kernel_code(self) -> str:
        '''
        Get CUDA code of the kernel
        '''

    @abc.abstractmethod
    def blocks_per_grid(self) -> list[int]:
        '''
        Get launch config: number of blocks per grid
        '''

    @abc.abstractmethod
    def threads_per_block(self) -> list[int]:
        '''
        Get launch config: number of threads per block
        '''

    @abc.abstractmethod
    def calc_target_outputs(self):
        '''
        Calculate target outputs using loaded / generated inputs
        '''


class KernelInterface(abc.ABC):

    def __init__(
        self, unique_id: str, kernel_code: str, shape: dict[str, Any],
        threads_per_block: list[int], blocks_per_grid: list[int],
        inputs: Iterable[_Tensor], outputs: Iterable[_Tensor]
    ):
        self._id = unique_id
        self._config = copy.deepcopy(shape)
        kernel_name = kernel_code[kernel_code.find('__global__ void') + 15:]
        kernel_name = kernel_name[:kernel_name.find('(')].strip()
        input_desc_list = [desc for tensor in inputs for desc in self._load_tensor(tensor)]
        output_desc_list = []
        for tensor in outputs:
            for desc in self._load_tensor(tensor):
                if desc['role'] == 'data':
                    output_desc_list.append(desc)
                else:
                    input_desc_list.append(desc)
        self._config |= {
            'MODULE_NAME': unique_id,
            'KERNEL_FUNC_NAME': kernel_name,
            'KERNEL_FUNC_BODY': kernel_code,
            'DIM_BLOCK': threads_per_block,
            'DIM_GRID': blocks_per_grid,
            'INPUTS': input_desc_list,
            'OUTPUTS': output_desc_list,
        }
        self._build()

    def _load_tensor(self, tensor: _Tensor) -> Iterable[dict]:
        if tensor.layout == 'dense':
            return [{
                'name': tensor.name,
                'type': tensor.dtype,
                'role': 'data',
                'shape': tensor.shape,
            }]
        else:
            sparse_desc = {}
            for k, v in tensor.sparse_desc().items():
                sparse_desc[k] = copy.deepcopy(v)
                if sparse_desc[k]['role'] == 'data':
                    sparse_desc[k]['type'] = tensor.dtype
                sparse_desc[k]['name'] = f'{tensor.name}_{k}'
            if tensor.dense_data is not None or tensor.sparse_data is not None:
                for k, v in tensor.sparse().items():
                    sparse_desc[k]['shape'] = v.shape
                    if sparse_desc[k]['role'] == 'tesa':
                        sparse_desc[k]['val'] = v.tolist()
            return sparse_desc.values()

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
        self._dir = os.path.join('tmp', self._id)
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        self._specify_data_path(self._config['INPUTS'])
        self._specify_data_path(self._config['OUTPUTS'])
        with open(os.path.join(TEMPLATE_DIR, 'test.cu.j2')) as f:
            self._template = f.read()
        self._code_path = os.path.join(self._dir, f'test.cu')
        self._exec_path = os.path.join(self._dir, 'test')

    def _build_exe(self):
        with open(self._code_path, 'w') as f:
            f.write(jinja2.Template(self._template).render(self._config))
        gpu_code = utils.cuda_detect()[0][1]
        build_args = f'arch=compute_{gpu_code},code=sm_{gpu_code}'
        self._run_cmd(
            f"nvcc -gencode {build_args} {self._code_path} -w -o {self._exec_path}",
            timeout=5
        )

    def _specify_data_path(self, desc_list: dict):
        for desc in desc_list:
            desc['filepath'] = os.path.join(self._dir, f'{desc["name"]}.dat')

    def _save_tensor(self, tensor: _Tensor):
        if tensor.layout == 'dense':
            with open(os.path.join(self._dir, f'{tensor.name}.dat'), 'wb') as f:
                tensor.dense().flatten().tofile(f)
        else:
            for k, v in tensor.sparse().items():
                with open(os.path.join(self._dir, f'{tensor.name}_{k}.dat'), 'wb') as f:
                    v.flatten().tofile(f)

    def __call__(
        self, inputs: Iterable[_Tensor], target_outputs: Optional[Iterable[_Tensor]] = None,
        num_warmups: int = 10, num_iters: int = 10, check_results: bool = True
    ) -> float:
        for tensor in inputs:
            self._save_tensor(tensor)
        if target_outputs is not None:
            for tensor in target_outputs:
                self._save_tensor(tensor)
        self._build_exe()
        result = self._run_cmd(
            f'{self._exec_path} {num_warmups} {num_iters} {int(check_results)}',
            timeout = 1 + 0.01 * num_iters
        )
        return float(result)

    def __del__(self):
        for file in glob.glob(f'{self._dir}/*'):
            os.remove(file)
        # shutil.rmtree(self._dir, ignore_errors=True)


class ModuleInterface(KernelInterface):

    def _build(self):
        self._module = cpp_extension.load_inline(
            self._id,
            '',
            cuda_sources=self.get_module_code(),
            extra_cflags=['-O3'],
        )

    def get_module_code(self) -> str:
        with open(os.path.join(TEMPLATE_DIR, 'module.cu.j2')) as f:
            module_template = f.read()
        return jinja2.Template(module_template).render(self._config)

    def get_module(self) -> torch.nn.Module:
        return self._module
