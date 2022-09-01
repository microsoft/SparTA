from dataclasses import dataclass
from abc import abstractmethod
from enum import Enum
from typing import Any, Union, Optional
import os
import jinja2

import numpy as np
import torch

if torch.cuda.is_available():
    # we may need to dry run without GPU (e.g., for document generation)
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

from sparta.common import tesa

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
    
    def changeable(self):
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
        self._declare_parameters()

    #################### parameters ####################
    def _add_parameter(self, p: _Parameter):
        self.parameters[p.name] = p

    def _set_parameter(self, name, value, force: bool=False):
        if self.parameters[name].value == value:
            return
        assert force or self.parameters[name].changeable(), f'{name} is not changeable' 
        self.parameters[name].value = value

    def set_parameters(self, dic: dict):
        for name, value in dic.items():
            if name in self.parameters:
                self._set_parameter(name, value)
        self._post_set_parameters()

    @abstractmethod
    def _declare_parameters(self):
        pass

    @abstractmethod
    def _post_set_parameters(self):
        pass

    def get_parameters(self):
        return {k:v.value for k,v in self.parameters.items()}

    def get_parameter(self, name):
        return self.parameters[name].value

    @abstractmethod
    def validate_parameters(self):
        return True

    #################### tuning related ####################
    def get_search_space(self):
        if self.search_space is not None:
            return self.search_space
        self.search_space = {k:v.default_space for k,v in self.parameters.items() if v.mode ==_Parameter._ParameterMode.tunable}

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

    def get_kernel_code(self):
        template_folder = os.path.join('sparta', 'specializer', 'jit', 'templates')
        fname = f'{self.template_name}.cuh.j2'
        with open(os.path.join(template_folder, fname)) as fn:
            src = fn.read()
        return jinja2.Template(src).render(self.get_parameters())

    def compile(self, sm_opts: dict = None):
        '''
        sm_opts: SourceModule options
        '''
        try:
            src = self.get_kernel_code()
            sm_opts = sm_opts or dict(options=['-O3'])
            self.kernel_func_call = SourceModule(src, **sm_opts).get_function(self.kernel_func_name)
        except Exception as err:
            print(err)
            print(src)
            raise RuntimeError from err

class MatMulKernelBase(TemplateKernelBase):
    __supported_modes__: str = ['dsd']

    def _declare_parameters(self):
        super()._declare_parameters()
        self._add_parameter(_Parameter(name="GLOBAL_M_VALUE"))
        self._add_parameter(_Parameter(name="GLOBAL_N_VALUE"))
        self._add_parameter(_Parameter(name="GLOBAL_K_VALUE"))
        self._add_parameter(_Parameter(name="BIASED"))
        self._add_parameter(_Parameter(name="TRANSPOSE"))

        self._add_parameter(_Parameter(name="BLOCK_SIZE_M_VALUE" , mode=_Parameter._ParameterMode.tunable, default_space=('choice', [8, 16, 32, 64, 128, 256])))
        self._add_parameter(_Parameter(name="BLOCK_SIZE_N_VALUE" , mode=_Parameter._ParameterMode.tunable, default_space=('choice', [8, 16, 32, 64, 128, 256])))
        self._add_parameter(_Parameter(name="BLOCK_SIZE_K_VALUE" , mode=_Parameter._ParameterMode.tunable, default_space=('choice', [8, 16, 32, 64, 128, 256])))

        # add inputs and outputs
        self.add_port('A', 'input', _Expr.from_str(["GLOBAL_M_VALUE", "GLOBAL_K_VALUE"]), 'float', 'dense')
        self.add_port('B', 'input', _Expr.from_str(["GLOBAL_K_VALUE", "GLOBAL_K_VALUE"]), 'float', 'dense')
        self.add_port('C', 'output', _Expr.from_str(["GLOBAL_M_VALUE", "GLOBAL_N_VALUE"]), 'float', 'dense', 'A @ B')

    def __init__(self, mode: str, transpose: bool=False, bias: bool=False):
        assert mode in self.__supported_modes__
        self.mode = mode
        super().__init__()
        self._set_parameter('BIASED', bias)
        self._set_parameter('TRANSPOSE', transpose)

        if mode == 'dsd':
            if bias:
                setattr(self, 'matmul', getattr(self, '__dsd_bias_call__'))
            else:
                setattr(self, 'matmul', getattr(self, '__dsd_call__'))
            self.get_port('B').to_sparse()
        else:
            raise NotImplementedError

    def _post_set_parameters(self):
        super()._post_set_parameters()
        self.mnk = (
            np.int32(self.get_parameter('GLOBAL_M_VALUE')), 
            np.int32(self.get_parameter('GLOBAL_N_VALUE')), 
            np.int32(self.get_parameter('GLOBAL_K_VALUE')), 
        )
        if self.mode == 'dsd':
            blk = (self.get_parameter('BLOCK_SIZE_N_VALUE'), self.get_parameter('BLOCK_SIZE_K_VALUE'))
            self.converter = tesa.BCSRObj('H', blk if self.get_parameter('TRANSPOSE') else (blk[1], blk[0]))


    @abstractmethod
    def __dsd_call__(self, *args):
        pass

    @abstractmethod
    def __dsd_bias_call__(self, *args):
        pass


class SparseMatMul(MatMulKernelBase):
    '''Template based sparse matmul kernel'''

    def _declare_parameters(self):
        super()._declare_parameters()

        self.template_name = f'sparse_matmul_{self.mode}'
        self.kernel_func_name = 'BLOCK_SPARSE_MATMUL'

        # add tunable parameters
        self._add_parameter(_Parameter(name="THREAD_SIZE_M_VALUE", mode=_Parameter._ParameterMode.tunable, default_space=('choice', [2, 4, 8, 16, 32])))
        self._add_parameter(_Parameter(name="THREAD_SIZE_N_VALUE", mode=_Parameter._ParameterMode.tunable, default_space=('choice', [2, 4, 8, 16, 32])))
        self._add_parameter(_Parameter(name="THREAD_SIZE_K_VALUE", mode=_Parameter._ParameterMode.tunable, default_space=('choice', [2, 4, 8, 16, 32])))

    def _post_set_parameters(self):
        super()._post_set_parameters()
        threads_per_block = (
            self.get_parameter('BLOCK_SIZE_N_VALUE') // self.get_parameter('THREAD_SIZE_N_VALUE'), 
            self.get_parameter('BLOCK_SIZE_M_VALUE') // self.get_parameter('THREAD_SIZE_M_VALUE'), 
            1)
        blocks_per_grid = (
            self.get_parameter('GLOBAL_N_VALUE') // self.get_parameter('BLOCK_SIZE_N_VALUE'), 
            self.get_parameter('GLOBAL_M_VALUE') // self.get_parameter('BLOCK_SIZE_M_VALUE'), 
            1)
        self.grid = _GridCfg(threads_per_block, blocks_per_grid)

    def __dsd_call__(self, A, VAL, PTR, IDX, C, bias=None):
        self.kernel_func_call(
            A, VAL, PTR, IDX, C, *self.mnk,
            block=self.grid.threads_per_block, grid=self.grid.blocks_per_grid)

    def __dsd_bias_call__(self, A, VAL, PTR, IDX, C, bias):
        self.kernel_func_call(
            A, VAL, PTR, IDX, C, bias, *self.mnk,
            block=self.grid.threads_per_block, grid=self.grid.blocks_per_grid)


class SparseMatMulOAI(MatMulKernelBase):
    pass


class KernelTuner:
    '''support kernel tuning with  multiple implementations'''

    def __init__(self, name: str, implements: dict[KernelBase], search_space: dict = None, backend: str = 'hyperopt') -> None:
        self.name = name
        self.backend = backend
        self.implements: dict[KernelBase] = implements
        self.best_kernel: KernelBase = None
        if self.backend == 'hyperopt':
            self.search_space = search_space or self.__hyperopt_create_search_space__()
        else:
            raise NotImplementedError

    def load_best_config(self, name: str, config: dict):
        self.best_kernel = self.implements[name]
        self.best_kernel.set_parameters(config)
        self.best_kernel.compile()

    def find_best_config(self, test_func: callable, algo: str, max_trials: int = None):
        if self.backend == 'hyperopt':
            return self.__hyperopt_tune__(test_func, algo, max_trials)
        raise NotImplementedError

    def __hyperopt_tune__(self, test_func: callable, algo: str, max_trials: int = None):
        from hyperopt import fmin, STATUS_FAIL, STATUS_OK, rand, tpe
        __algo__ = {'random': rand, 'tpe': tpe.suggest}
        assert algo in __algo__

        def _objective(args: dict):
            kern = self.implements[args['kernel']]
            kern.set_parameters(args)
            if kern.validate_parameters():
                return {'loss': test_func(kern), 'status': STATUS_OK}
            return {'status': STATUS_FAIL}

        return fmin(
            _objective, self.search_space, algo=__algo__[algo], max_evals=max_trials
            )

    def __hyperopt_create_search_space__(self):
        from hyperopt import hp
        choices = []
        for kname, kern in self.implements.items():
            space = {'kernel': kname}
            for param, v in kern.get_search_space().items():
                if v[0] == 'choice':
                    space[param] = hp.choice(f'{self.name}::{param}', v[1])
                else:
                    raise NotImplementedError
            choices.append(space)
        return hp.choice(self.name, choices)        


class SparseOpBase:
    def __init__(self, dense_op: torch.nn.Linear, config: dict, keep_origin_op: bool=True) -> None:
        self.origin_op = dense_op if keep_origin_op else None
        weight = dense_op.weight.clone().detach()
        bias = None if dense_op.bias is None else dense_op.bias.clone().detach()
        N, K = weight.shape