# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import abc
import copy
import shutil
import hashlib
import subprocess
from typing import Any, Optional, Callable, Iterable, Union, Tuple, List, Dict
from dataclasses import dataclass

import torch
import jinja2
import numpy as np

from sparta import __env_ready__
if __env_ready__:
    # we may need to dry run without GPU (e.g., for document generation)
    from torch.utils import cpp_extension
    import pycuda.autoprimaryctx
    from pycuda.compiler import SourceModule


from sparta.common import tesa, utils
from sparta.common.tuning import TunableItemCfg


TEMPLATE_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], "templates")


@dataclass
class _Parameter:
    name: str
    value: Any
    is_tunable: Optional[bool] = False
    is_dynamic: Optional[bool] = False
    search_space: Optional[TunableItemCfg] = None

    def __post_init__(self):
        if self.search_space is not None:
            assert self.is_tunable


@dataclass
class _Tensor:
    name: str
    dtype: str
    layout: str
    layout_parent: Optional['_Tensor'] = None
    default_val: Optional[Union[float, int]] = None
    shape: Optional[Tuple[str]] = None
    mask: Optional[np.ndarray] = None
    layout_config: Optional[Dict] = None
    dense_data: Optional[np.ndarray] = None
    sparse_data: Optional[Dict[str, np.ndarray]] = None

    def set_data(self, data: np.ndarray, auto_mask: bool = True):
        assert data.shape == self.shape
        self.dense_data = data
        self.sparse_data = None
        if self.mask is not None and auto_mask:
            self.dense_data *= self.mask

    def generate_data(self):
        assert self.shape is not None
        if self.dense_data is None:
            if self.default_val is not None:
                self.dense_data = np.zeros(shape=self.shape) + self.default_val
                self.dense_data = self.dense_data.astype(f'{self.dtype}32')
            else:
                self.dense_data = np.random.uniform(size=self.shape).astype(f'{self.dtype}32')
        if self.layout != 'dense':
            if self.mask is None:
                self.generate_mask()

    def generate_mask(self, sparsity: float = 0.8):
        assert self.shape is not None
        assert self.layout_config is not None
        if self.layout == 'BCSR':
            if self.layout_parent is None:
                block_size = self.layout_config['block_size']
                row_num = self.shape[-2] // block_size[0]
                col_num = self.shape[-1] // block_size[1]
                mask = np.random.uniform(size=(row_num, col_num)) > sparsity
                mask = np.tile(mask.reshape((row_num, col_num, 1, 1)), [1, 1] + block_size)
                self.mask = mask.swapaxes(1, 2).reshape(self.shape)
            else:
                self.mask = self.layout_parent.mask
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
                size=self.shape,
                mask=self.mask,
                **self.sparse_data,
                **self.layout_config
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
                dense=self.dense_data,
                size=self.shape,
                mask=self.mask,
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
        self.parameters: Dict[str, _Parameter] = {}
        self.inputs: Dict[str, _Tensor] = {}
        self.outputs: Dict[str, _Tensor] = {}
        self.add_parameters()
        self.add_ports()

    def add_parameter(
        self, name: str, value: Any = None, is_tunable: bool = False, is_dynamic: bool = False,
        search_space: Optional[List[Any]] = None
    ):
        self.parameters[name] = _Parameter(name, value, is_tunable, is_dynamic, search_space)

    def set_search_space(self, search_space: Dict[str, List[Any]]):
        for name, space in search_space.items():
            self.parameters[name].search_space = space

    def get_search_space(self):
        return {p.name: p.search_space for p in self.parameters.values() if p.is_tunable}

    def set_parameter(self, name, value):
        if name not in self.parameters and name in ['_name']:
            return  # ignore some special key words
        self.parameters[name].value = value

    def set_parameters(self, dic: dict):
        for name, value in dic.items():
            self.set_parameter(name, value)

    def get_parameter(self, name):
        return self.parameters[name].value

    def get_parameters(self, names: list = None):
        if names is None:
            return {k: v.value for k, v in self.parameters.items()}
        else:
            return {k: self.parameters[k].value for k in names}

    def add_input(
        self, name: str, dtype: str, layout: str = 'dense',
        default_val: Optional[Union[int, float]] = None
    ):
        self.inputs[name] = _Tensor(name, dtype, layout, default_val=default_val)

    def set_input_shape(self, name: str, shape: Tuple[str]):
        self.inputs[name].shape = shape
        self.inputs[name].dense_data = None

    def set_input_layout(self, name: str, layout: Union[Dict, _Tensor]):
        if isinstance(layout, _Tensor):
            self.inputs[name].layout_config = layout.layout_config
            self.inputs[name].layout_parent = layout
        else:
            self.inputs[name].layout_config = layout
        self.inputs[name].sparse_data = None

    def set_input(self, name: str, data: np.ndarray):
        self.inputs[name].set_data(data)

    def get_input(self, name: str):
        return self.inputs[name]

    def add_output(self, name: str, dtype: str, layout: str = 'dense'):
        self.outputs[name] = _Tensor(name, dtype, layout)

    def set_output_shape(self, name: str, shape: Tuple[str]):
        self.outputs[name].shape = shape

    def set_output_layout(self, name: str, layout: Union[Dict, _Tensor]):
        if isinstance(layout, _Tensor):
            self.outputs[name].layout_config = layout.layout_config
            self.outputs[name].layout_parent = layout
        else:
            self.outputs[name].layout_config = layout

    def set_target_output(self, name: str, data: np.ndarray, auto_mask: bool = True):
        self.outputs[name].set_data(data, auto_mask)

    def get_output(self, name: str):
        return self.outputs[name]

    def set_mask(
        self, mask: Optional[Dict[str, np.ndarray]] = None,
        generate_if_missing: bool = True
    ):
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
        self, config: Dict, mask: Optional[Dict[str, np.ndarray]],
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
        self, config: Dict, mask: Optional[Dict[str, np.ndarray]] = None,
        inputs: Dict[str, np.ndarray] = None, target_outputs: Dict[str, np.ndarray] = None,
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
            unique_id=unique_id,
            kernel_code=self.get_kernel_code(),
            shape=config,
            threads_per_block=self.threads_per_block(),
            blocks_per_grid=self.blocks_per_grid(),
            inputs=self.inputs.values(),
            outputs=self.outputs.values()
        )
        lat = test_func(test_inputs, test_outputs, num_warmups, num_iters, check_results)
        return lat

    def compile(self, config: Dict, mask: Dict[str, np.ndarray], jit: bool = True):
        unique_id = self.configure(config, mask, False)
        for input_tensor in self.inputs.values():
            input_tensor.generate_data()
        for output_tensor in self.outputs.values():
            output_tensor.generate_data()
        module_factory = JITInterface if jit else ModuleInterface
        return module_factory(
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
    def blocks_per_grid(self) -> Tuple[int]:
        '''
        Get launch config: number of blocks per grid
        '''

    @abc.abstractmethod
    def threads_per_block(self) -> Tuple[int]:
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
        self, unique_id: str, kernel_code: str, shape: Dict[str, Any],
        threads_per_block: Tuple[int], blocks_per_grid: Tuple[int],
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
        self._config.update({
            'MODULE_NAME': unique_id,
            'KERNEL_FUNC_NAME': kernel_name,
            'KERNEL_FUNC_BODY': kernel_code,
            'DIM_BLOCK': threads_per_block,
            'DIM_GRID': blocks_per_grid,
            'INPUTS': input_desc_list,
            'OUTPUTS': output_desc_list,
        })
        self._build()

    def _load_tensor(self, tensor: _Tensor) -> Iterable[Dict]:
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
            if tensor.layout_parent is not None:
                sparse_desc = {k: v for k, v in sparse_desc.items() if v['role'] == 'data'}
            return sparse_desc.values()

    def _run_cmd(self, cmd: str, timeout: float) -> str:
        process = subprocess.Popen(f'exec {cmd}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            process.kill()
            process.wait()
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
        self._dir = os.path.join('/tmp/sparta', self._id)
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

    def _specify_data_path(self, desc_list: Dict):
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
            timeout=1 + 0.01 * num_iters
        )
        shutil.rmtree(self._dir, ignore_errors=True)
        return float(result)


class ModuleInterface(KernelInterface):

    def _build(self):
        print('Building PyTorch Module, it will take about one minute...')
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


class JITInterface(KernelInterface):

    def _build(self):
        self._kernel_func_name = self._config['KERNEL_FUNC_NAME']
        self._kernel_func_body = self._config['KERNEL_FUNC_BODY']
        self._threads_per_block = self._config['DIM_BLOCK']
        self._blocks_per_grid = self._config['DIM_GRID']
        self._input_mask = []
        self._fixed_inputs = []
        for x in self._config['INPUTS']:
            if x['role'] == 'data':
                self._input_mask.append(True)
            else:
                self._input_mask.append(False)
                val = np.array(x['val']).astype(f'{x["type"]}32')
                self._fixed_inputs.append(torch.from_numpy(val))
        self._output_placeholder = []
        for y in self._config['OUTPUTS']:
            val = np.zeros(y['shape'], dtype=f'{y["type"]}32')
            self._output_placeholder.append(torch.from_numpy(val))

    def get_module(self) -> torch.nn.Module:
        return JITModule(
            self._kernel_func_name,
            self._kernel_func_body,
            self._blocks_per_grid,
            self._threads_per_block,
            self._input_mask,
            self._fixed_inputs,
            self._output_placeholder,
        )


class JITModule(torch.nn.Module):

    def __init__(
        self, kernel_func_name: str, kernel_func_body: str,
        blocks_per_grid: Tuple[int], threads_per_block: Tuple[int],
        input_mask: List[bool], fixed_inputs: List[torch.Tensor],
        output_placeholder: List[torch.Tensor]
    ):
        super().__init__()
        params = [torch.nn.Parameter(x, requires_grad=False) for x in fixed_inputs]
        self._params = torch.nn.ParameterList(params).cuda()
        source_module = SourceModule(kernel_func_body, options=['-O3'])
        self._kernel_func_call = source_module.get_function(kernel_func_name)
        self._blocks_per_grid = blocks_per_grid + tuple(1 for _ in range(3 - len(blocks_per_grid)))
        self._threads_per_block = threads_per_block + tuple(1 for _ in range(3 - len(threads_per_block)))
        self._input_mask = input_mask
        self._outputs = [y.cuda() for y in output_placeholder]
        self.func_name = kernel_func_name
        self.func_body = kernel_func_body

    def forward(self, *args):
        inputs = []
        arg_idx = 0
        param_idx = 0
        for is_arg in self._input_mask:
            if is_arg:
                inputs.append(args[arg_idx])
                arg_idx += 1
            else:
                inputs.append(self._params[param_idx])
                param_idx += 1
        self._kernel_func_call(
            *inputs, *(self._outputs),
            block=self._threads_per_block,
            grid=self._blocks_per_grid
        )
        if len(self._outputs) == 1:
            return self._outputs[0]
        else:
            return self._outputs
