from dataclasses import dataclass
from typing import Any, Union, Optional
from ast import literal_eval
import os
import jinja2

import numpy as np
import torch

# import cutex
from pycuda.compiler import SourceModule


@dataclass
class _Parameter:
    name: str
    value: Any
    is_tuable: bool
    search_space: list[Any]

    def __post_init__(self):
        if self.search_space is not None:
            assert self.is_tuable


@dataclass
class _Expr:
    tokens: list[str]

    @staticmethod
    def from_str(s: Union[str,tuple,list]):
        if isinstance(s, str):
            return _Expr(s.split(' '))
        elif isinstance(s, tuple):
            return (_Expr(ss.split(' ')) for ss in s)
        elif isinstance(s, list):
            return [_Expr(ss.split(' ')) for ss in s]
        raise ValueError


@dataclass
class _Tensor:
    name: str
    direction: str
    shape: list[str,_Expr]
    dtype: str
    layout: Union[str,dict]
    formula: str = None

    def __post_init__(self):
        assert self.direction in ['input', 'output']

    def to_sparse(self):
        pass


@dataclass
class _GridCfg:
    threads_per_block: tuple
    blocks_per_grid: tuple

    def __post_init__(self):
        assert len(self.threads_per_block) == 3


class KernelBase:

    def __init__(self):
        self.parameters: dict[_Parameter] = {}
        self.ports: list = []
        self.grid: _GridCfg = None

    def add_parameter(self, name: str, value: Any = None, is_tunable: bool=False, search_space = None):
        self.parameters[name] = _Parameter(name, value, is_tunable, search_space)

    def set_parameter(self, name, value):
        self.parameters[name].value = value

    def set_parameters(self, dic: dict):
        for name, value in dic.items():
            self.parameters[name].value = value

    def pre_kernel_launch(self):
        pass

    def get_parameters(self):
        return {k:v.value for k,v in self.parameters.items()}

    def get_parameter(self, name):
        return self.parameters[name].value

    def add_port(self, name: str, direction: str, shape: list, dtype:str, layout: Union[str,dict]='dense', formula: str=None):
        self.ports.append(_Tensor(name, direction, shape, dtype, layout, formula))

    def get_port(self, name: str):
        rst = [p for p in self.ports if p.name==name]
        return rst[0] if rst else None

    def expr_eval(self, e: _Expr):
        dic = self.get_parameters()
        s = ' '.join([str(dic[t]) if t in dic else t for t in e.tokens])
        return eval(s)


class TemplateKernelBase(KernelBase):

    def __init__(self):
        super().__init__()
        self.template_name = None
        self.kernel_func_name = None
        self.kernel_func_call = None

    def get_kernel_code(self, params: Optional[dict]):
        if params is not None:
            self.set_parameters(params)
        template_folder = os.path.join('sparta', 'specializer', 'jit', 'templates')
        fname = f'{self.template_name}.cuh.j2'
        with open(os.path.join(template_folder, fname)) as fn:
            src = fn.read()
        return jinja2.Template(src).render(self.get_parameters())

    def compile(self, params: Optional[dict]=None):
        src = self.get_kernel_code(params)
        # kernels = cutex.SourceModule(src, options=['-O3'])
        # self.kernel_func_call = getattr(kernels, self.kernel_func_name)
        self.kernel_func_call = SourceModule(src, options=['-O3']).get_function(self.kernel_func_name)
        self.pre_kernel_launch()


class MatMulKernelBase(TemplateKernelBase):

    def __init__(self):
        super().__init__()

        # add parameters
        self.add_parameter("GLOBAL_M_VALUE")
        self.add_parameter("GLOBAL_N_VALUE")
        self.add_parameter("GLOBAL_K_VALUE")
        self.add_parameter("BIASED")
        self.add_parameter("TRANSPOSE")

        # add inputs and outputs
        self.add_port('A', 'input', _Expr.from_str(["GLOBAL_M_VALUE", "GLOBAL_K_VALUE"]), 'float', 'dense')
        self.add_port('B', 'input', _Expr.from_str(["GLOBAL_K_VALUE", "GLOBAL_K_VALUE"]), 'float', 'dense')
        self.add_port('C', 'output', _Expr.from_str(["GLOBAL_M_VALUE", "GLOBAL_N_VALUE"]), 'float', 'dense', 'A @ B')

        # add tunable parameters
        self.add_parameter("BLOCK_SIZE_M_VALUE" , is_tunable=True, search_space=[])
        self.add_parameter("BLOCK_SIZE_N_VALUE" , is_tunable=True, search_space=[])
        self.add_parameter("BLOCK_SIZE_K_VALUE" , is_tunable=True, search_space=[])
        self.add_parameter("THREAD_SIZE_M_VALUE", is_tunable=True, search_space=[])
        self.add_parameter("THREAD_SIZE_N_VALUE", is_tunable=True, search_space=[])
        self.add_parameter("THREAD_SIZE_K_VALUE", is_tunable=True, search_space=[])

        # kernel launching config
        self.threads_per_block = _Expr.from_str(["BLOCK_SIZE_N_VALUE // THREAD_SIZE_N_VALUE", "BLOCK_SIZE_M_VALUE // THREAD_SIZE_M_VALUE"])
        self.blocks_per_grid = _Expr.from_str(["GLOBAL_N_VALUE // BLOCK_SIZE_N_VALUE", "GLOBAL_M_VALUE // BLOCK_SIZE_M_VALUE"])

    def pre_kernel_launch(self):
        threads_per_block = (self.parameters['BLOCK_SIZE_N_VALUE'].value // self.parameters['THREAD_SIZE_N_VALUE'].value, self.parameters['BLOCK_SIZE_M_VALUE'].value // self.parameters['THREAD_SIZE_M_VALUE'].value, 1)
        blocks_per_grid = (self.parameters['GLOBAL_N_VALUE'].value // self.parameters['BLOCK_SIZE_N_VALUE'].value, self.parameters['GLOBAL_M_VALUE'].value // self.parameters['BLOCK_SIZE_M_VALUE'].value, 1)
        self.grid = _GridCfg(threads_per_block, blocks_per_grid)


class SparseMatMul(MatMulKernelBase):

    def __init__(self, mode: str, transpose: bool=False, bias: bool=False):
        super().__init__()

        self.set_parameter('TRANSPOSE', transpose)
        self.set_parameter('BIASED', bias)

        self.mode = mode
        if mode == 'dsd':
            self.template_name = f'sparse_matmul_{mode}'
            self.kernel_func_name = 'BLOCK_SPARSE_MATMUL'
            if bias:
                setattr(self, 'matmul', getattr(self, '__dsd_bias_call__'))
            else:
                setattr(self, 'matmul', getattr(self, '__dsd_call__'))
            self.get_port('B').to_sparse()

    def __dsd_call__(self, A, VAL, PTR, IDX, C, bias=None):
        self.kernel_func_call(A, VAL, PTR, IDX, C, block=self.grid.threads_per_block, grid=self.grid.blocks_per_grid)

    def __dsd_bias_call__(self, A, VAL, PTR, IDX, C, bias):
        self.kernel_func_call(A, VAL, PTR, IDX, C, bias, block=self.grid.threads_per_block, grid=self.grid.blocks_per_grid)

    # def __call__(self, A:torch.Tensor, B: dict, C: torch.Tensor, bias=None):
        # self.kernel_func_call(A, B['val'], B['row_ptr'], B['col_idx'], C, block=self.grid.threads_per_block, grid=self.grid.blocks_per_grid)