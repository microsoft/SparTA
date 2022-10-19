import abc
import warnings
import dataclasses
from typing import Any, Dict, List, Tuple, Callable, Optional

import torch

from sparta import __env_ready__
if __env_ready__:
    # we may need to dry run without GPU (e.g., for document generation)
    import pycuda.autoprimaryctx
    from pycuda.compiler import SourceModule

from sparta.common.tesa import TeSAConverter
from sparta.common.tuning import TunableItemCfg

@dataclasses.dataclass
class _Parameter:
    name: str
    value: Any
    is_tunable: Optional[bool] = False
    is_dynamic: Optional[bool] = False
    search_space: Optional[TunableItemCfg] = None

    def __post_init__(self):
        if self.search_space is not None:
            assert self.is_tunable


class KernelBase(Callable):

    def __init__(self):
        self._parameters: Dict[str, _Parameter] = {}
        self._converters: Dict[str, TeSAConverter] = {}
        self._masks: Dict[str, torch.Tensor] = {}
        self._func: Callable = None
        self.tesa_type: Dict[str, type[TeSAConverter]] = {}
        self.tesa_attrs: Dict[str, List[str]] = {}
        self.ready = False
        self.add_parameters()
        self.set_tesa()

    @abc.abstractmethod
    def set_tesa(self):
        '''Set TeSA types and attrs of sparse tensors.'''

    @abc.abstractmethod
    def add_parameters(self):
        '''Add kernel-specialized parameters.'''
    
    def add_parameter(
        self, name: str, value: Any = None, is_tunable: bool = False, is_dynamic: bool = False,
        search_space: Optional[List[Any]] = None
    ):
        self._parameters[name] = _Parameter(name, value, is_tunable, is_dynamic, search_space)

    def set_search_space(self, search_space: Dict[str, List[Any]]):
        for name, space in search_space.items():
            self._parameters[name].search_space = space

    def get_search_space(self):
        return {p.name: p.search_space for p in self._parameters.values() if p.is_tunable}

    def set_parameter(self, name: str, value: Any):
        if name not in self._parameters and name in ['_name']:
            return  # ignore some special key words
        self._parameters[name].value = value

    def set_parameters(self, dic: Dict[str, Any]):
        for name, value in dic.items():
            self.set_parameter(name, value)

    def get_parameter(self, name: str):
        return self._parameters[name].value

    def get_parameters(self, names: Optional[List[str]] = None):
        if names is None:
            return {k: v.value for k, v in self._parameters.items()}
        else:
            return {k: self._parameters[k].value for k in names}

    def set_mask(self, name: str, value: torch.Tensor):
        self._masks[name] = value

    def set_masks(self, mask_dict: Dict[str, torch.Tensor]):
        for name, value in mask_dict.items():
            self.set_mask(name, value)

    def get_mask(self, name: str):
        return self._masks[name]

    def set_converter(self, name: str, converter: TeSAConverter):
        self._converters[name] = converter

    def get_converter(self, name: str):
        return self._converters[name]

    @abc.abstractmethod
    def set_shape(self, *args, **kwargs):
        '''Set shape parameters.'''

    @abc.abstractmethod
    def get_kernel_code(self) -> str:
        '''Get CUDA code of the kernel.'''

    @abc.abstractmethod
    def blocks_per_grid(self: int) -> Tuple[int]:
        '''Get launch config: number of blocks per grid.'''

    @abc.abstractmethod
    def threads_per_block(self) -> Tuple[int]:
        '''Get launch config: number of threads per block.'''

    @abc.abstractmethod
    def pre_compile(self) -> Tuple[List[bool], List[torch.Tensor], List[Tuple]]:
        '''Calc input_mask, fixed_inputs and output_shapes.'''

    def compile(self, config: Dict[str, Any], mask: Dict[str, torch.Tensor]):
        self.set_parameters(config)
        self.set_masks(mask)
        kernel_code = self.get_kernel_code()
        kernel_name = kernel_code[kernel_code.find('__global__ void') + 15:]
        kernel_name = kernel_name[:kernel_name.find('(')].strip()

        input_mask, fixed_inputs, output_shapes = self.pre_compile()

        self._func = JITModule(
            kernel_func_name=kernel_name,
            kernel_func_body=kernel_code,
            blocks_per_grid=self.blocks_per_grid(),
            threads_per_block=self.threads_per_block(),
            input_mask=input_mask,
            fixed_inputs=fixed_inputs,
            output_shapes=output_shapes
        ).forward
        self.ready = True

    def __call__(self, *args) -> torch.Tensor:
        if self.ready:
            return self._func(*args)
        else:
            raise ValueError('The kernel is not compiled.')


class JITModule(torch.nn.Module):

    def __init__(
        self, kernel_func_name: str, kernel_func_body: str,
        blocks_per_grid: Tuple[int], threads_per_block: Tuple[int],
        input_mask: List[bool], fixed_inputs: List[torch.Tensor],
        output_shapes: List[Tuple[int]]
    ):
        super().__init__()
        params = [torch.nn.Parameter(x, requires_grad=False) for x in fixed_inputs]
        self._params = torch.nn.ParameterList(params).cuda()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            source_module = SourceModule(kernel_func_body, options=['-O3'])
        self._kernel_func_call = source_module.get_function(kernel_func_name)

        def fill_launch_config(x: Tuple[int]):
            dims = len(x)
            if dims < 3:
                return x + tuple(1 for _ in range(3 - len(x)))
            else:
                return x[:3]

        self._blocks_per_grid = fill_launch_config(blocks_per_grid)
        self._threads_per_block = fill_launch_config(threads_per_block)
        self._input_mask = input_mask
        self._outputs = [torch.zeros(shape).cuda() for shape in output_shapes]
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
