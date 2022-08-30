from dataclasses import dataclass
from enum import Enum
from typing import Any, Union, Optional
import os
import jinja2

import numpy as np
import torch

# import cutex
import pycuda.autoinit
from pycuda.compiler import SourceModule


@dataclass
class _Parameter:

    class _ParameterMode(Enum):
        init_only = 1
        mutable = 2
        tunable = 3

    name: str
    value: Any = None
    description: str = None
    mode: _ParameterMode = _ParameterMode.init_only 
    # for tunable parameter only
    default_space: tuple = None # (type, options), type in ['choice']

    def __post_init__(self):
        if self.default_space is not None:
            assert self.mode == _Parameter._ParameterMode.tunable
            space_type, space_opts = self.default_space
            if space_type == 'choice':
                assert isinstance(space_opts, list)
            else:
                raise ValueError
    
    def is_changable(self):
        return self.mode != _Parameter._ParameterMode.init_only or self.value is None


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
        self.search_space: dict = None

    #################### parameters ####################
    def _add_parameter(self, p: _Parameter):
        self.parameters[p.name] = p

    def _set_parameter(self, name, value, force: bool=False):
        if self.parameters[name].value == value:
            return
        assert force or self.parameters[name].is_changable(), f'{name} is not changable' 
        self.parameters[name].value = value

    def set_parameters(self, dic: dict):
        for name, value in dic.items():
            self._set_parameter(name, value)
        self._post_set_parameters()

    def _post_set_parameters(self):
        pass

    def get_parameters(self):
        return {k:v.value for k,v in self.parameters.items()}

    def get_parameter(self, name):
        return self.parameters[name].value

    #################### tuning related ####################
    def get_search_space(self):
        if self.search_space is not None:
            return self.search_space
        self.search_space = {k:v.default_space for k,v in self.parameters.items() if v.mode =='tunable'}

    #################### ports (inout) ####################
    def add_port(self, name: str, direction: str, shape: list, dtype:str, layout: Union[str,dict]='dense', formula: str=None):
        self.ports.append(_Tensor(name, direction, shape, dtype, layout, formula))

    def get_port(self, name: str):
        rst = [p for p in self.ports if p.name==name]
        return rst[0] if rst else None

    #################### process ####################

    #################### misc ####################
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
        try:
            self.kernel_func_call = SourceModule(src, options=['-O3']).get_function(self.kernel_func_name)
        except Exception as err:
            print(err)
            print(src)
            raise RuntimeError

class MatMulKernelBase(TemplateKernelBase):

    def __init__(self):
        super().__init__()

        # add parameters
        self._add_parameter(_Parameter(name="GLOBAL_M_VALUE"))
        self._add_parameter(_Parameter(name="GLOBAL_N_VALUE"))
        self._add_parameter(_Parameter(name="GLOBAL_K_VALUE"))
        self._add_parameter(_Parameter(name="BIASED"))
        self._add_parameter(_Parameter(name="TRANSPOSE"))

        # add inputs and outputs
        self.add_port('A', 'input', _Expr.from_str(["GLOBAL_M_VALUE", "GLOBAL_K_VALUE"]), 'float', 'dense')
        self.add_port('B', 'input', _Expr.from_str(["GLOBAL_K_VALUE", "GLOBAL_K_VALUE"]), 'float', 'dense')
        self.add_port('C', 'output', _Expr.from_str(["GLOBAL_M_VALUE", "GLOBAL_N_VALUE"]), 'float', 'dense', 'A @ B')

        # add tunable parameters
        self._add_parameter(_Parameter(name="BLOCK_SIZE_M_VALUE" , mode=_Parameter._ParameterMode.tunable, default_space=('choice', [8, 16, 32, 64, 128, 256])))
        self._add_parameter(_Parameter(name="BLOCK_SIZE_N_VALUE" , mode=_Parameter._ParameterMode.tunable, default_space=('choice', [8, 16, 32, 64, 128, 256])))
        self._add_parameter(_Parameter(name="BLOCK_SIZE_K_VALUE" , mode=_Parameter._ParameterMode.tunable, default_space=('choice', [8, 16, 32, 64, 128, 256])))
        self._add_parameter(_Parameter(name="THREAD_SIZE_M_VALUE", mode=_Parameter._ParameterMode.tunable, default_space=('choice', [2, 4, 8, 16, 32])))
        self._add_parameter(_Parameter(name="THREAD_SIZE_N_VALUE", mode=_Parameter._ParameterMode.tunable, default_space=('choice', [2, 4, 8, 16, 32])))
        self._add_parameter(_Parameter(name="THREAD_SIZE_K_VALUE", mode=_Parameter._ParameterMode.tunable, default_space=('choice', [2, 4, 8, 16, 32])))

        # kernel launching config
        self.threads_per_block = _Expr.from_str(["BLOCK_SIZE_N_VALUE // THREAD_SIZE_N_VALUE", "BLOCK_SIZE_M_VALUE // THREAD_SIZE_M_VALUE"])
        self.blocks_per_grid = _Expr.from_str(["GLOBAL_N_VALUE // BLOCK_SIZE_N_VALUE", "GLOBAL_M_VALUE // BLOCK_SIZE_M_VALUE"])

    def _post_set_parameters(self):
        threads_per_block = (self.parameters['BLOCK_SIZE_N_VALUE'].value // self.parameters['THREAD_SIZE_N_VALUE'].value, self.parameters['BLOCK_SIZE_M_VALUE'].value // self.parameters['THREAD_SIZE_M_VALUE'].value, 1)
        blocks_per_grid = (self.parameters['GLOBAL_N_VALUE'].value // self.parameters['BLOCK_SIZE_N_VALUE'].value, self.parameters['GLOBAL_M_VALUE'].value // self.parameters['BLOCK_SIZE_M_VALUE'].value, 1)
        self.grid = _GridCfg(threads_per_block, blocks_per_grid)
        self.mnk = (
            np.int32(self.get_parameter('GLOBAL_M_VALUE')), 
            np.int32(self.get_parameter('GLOBAL_N_VALUE')), 
            np.int32(self.get_parameter('GLOBAL_K_VALUE')), 
        )


class SparseMatMul(MatMulKernelBase):

    def __init__(self, mode: str, transpose: bool=False, bias: bool=False):
        super().__init__()

        self._set_parameter('TRANSPOSE', transpose, force=True)
        self._set_parameter('BIASED', bias, force=True)

        self.mode = mode
        if mode == 'dsd':
            self.template_name = f'sparse_matmul_{mode}'
            self.kernel_func_name = 'BLOCK_SPARSE_MATMUL'
            if bias:
                setattr(self, 'matmul', getattr(self, '__dsd_bias_call__'))
            else:
                setattr(self, 'matmul', getattr(self, '__dsd_call__'))
            self.get_port('B').to_sparse()
        else:
            raise NotImplementedError

    def __dsd_call__(self, A, VAL, PTR, IDX, C, bias=None):
        self.kernel_func_call(
            A, VAL, PTR, IDX, C, *self.mnk,
            block=self.grid.threads_per_block, grid=self.grid.blocks_per_grid)

    def __dsd_bias_call__(self, A, VAL, PTR, IDX, C, bias):
        self.kernel_func_call(
            A, VAL, PTR, IDX, C, bias, *self.mnk,
            block=self.grid.threads_per_block, grid=self.grid.blocks_per_grid)

    # def __call__(self, A:torch.Tensor, B: dict, C: torch.Tensor, bias=None):
        # self.kernel_func_call(A, B['val'], B['row_ptr'], B['col_idx'], C, block=self.grid.threads_per_block, grid=self.grid.blocks_per_grid)


class SparseOpBase:
    def __init__(self, dense_op: torch.nn.Module, config: dict) -> None:
        pass