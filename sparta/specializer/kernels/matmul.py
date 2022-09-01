# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import abc

import jinja2
import numpy as np

from sparta.specializer.kernels.kernel_base import KernelBase


TEMPLATE_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], "templates")

class MatMulKernelBase(KernelBase):

    def __init__(
        self, sparse_type: str, dtype: str = 'float',
        biased: bool = True, transpose: bool = True, compressed: bool = True
    ):
        if sparse_type not in ['sdd', 'dsd', 'dds']:
            raise ValueError(f'invalid sparse type: {sparse_type}')
        self._biased = biased
        self._transpose = transpose
        self._compressed = compressed
        self._stype = sparse_type
        self._dtype = dtype
        super().__init__()

    def get_kernel_name(self) -> str:
        b_str = '_b' if self._biased else ''
        t_str = '_t' if self._transpose else ''
        c_str = '_c' if self._compressed else ''
        return f'sparse_matmul_{self._stype}{b_str}{t_str}{c_str}'

    @abc.abstractmethod
    def get_kernel_code(self) -> str:
        '''
        Get CUDA code of the kernel
        '''

    def add_parameters(self):
        self.add_parameter("GLOBAL_M_VALUE", value=4096)
        self.add_parameter("GLOBAL_K_VALUE", value=768)
        self.add_parameter("GLOBAL_N_VALUE", value=3072)
        self.add_parameter("BIASED", value=self._biased)
        self.add_parameter("TRANSPOSE", value=self._transpose)
        self.add_parameter("COMPRESSED", value=self._compressed)

    @abc.abstractmethod
    def check_parameters(self):
        '''
        Check if parameters are valid
        '''

    def add_ports(self):
        self.add_input('A', self._dtype, 'BCSR' if self._stype == 'sdd' else 'dense')
        self.add_input('B', self._dtype, 'BCSR' if self._stype == 'dsd' else 'dense')
        if self._biased:
            self.add_input('bias', self._dtype, 'dense')
        self.add_output('C', self._dtype, 'BCSR' if self._stype == 'dds' else 'dense')

    @abc.abstractmethod
    def set_ports_shape(self):
        M = self.get_parameter('GLOBAL_M_VALUE')
        K = self.get_parameter('GLOBAL_K_VALUE')
        N = self.get_parameter('GLOBAL_N_VALUE')
        self.set_input_shape('A', (M, K))
        if self._transpose:
            self.set_input_shape('B', (N, K))
        else:
            self.set_input_shape('B', (K, N))
        if self._biased:
            self.set_input_shape('bias', (N, ))
        self.set_output_shape('C', (M, N))

    @abc.abstractmethod
    def set_ports_layout(self):
        '''
        Set layout configs of inputs and outputs using determined parameters
        '''

    @abc.abstractmethod
    def blocks_per_grid(self) -> tuple[int]:
        '''
        Get launch config: number of blocks per grid
        '''

    @abc.abstractmethod
    def threads_per_block(self) -> tuple[int]:
        '''
        Get launch config: number of threads per block
        '''

    def calc_target_outputs(self) -> dict[str, np.ndarray]:
        A = self.get_input('A').dense()
        B = self.get_input('B').dense()
        if self._transpose:
            B = B.T
        C = A @ B
        if self._biased:
            C += self.get_input('bias').dense()
        self.set_target_output('C', C)


class OurTemplateSparseMatMulKernel(MatMulKernelBase):

    def get_kernel_name(self) -> str:
        return f'our_{super().get_kernel_name()}'

    def get_kernel_code(self) -> str:
        with open(os.path.join(TEMPLATE_DIR, f'our_sparse_matmul_{self._stype}.cuh.j2')) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())

    def add_parameters(self):
        super().add_parameters()
        self.add_parameter("BLOCK_SIZE_M_VALUE" , is_tunable=True, search_space=[16, 32, 64])
        self.add_parameter("BLOCK_SIZE_N_VALUE" , is_tunable=True, search_space=[16, 32, 64])
        self.add_parameter("BLOCK_SIZE_K_VALUE" , is_tunable=True, search_space=[16, 32, 64])
        self.add_parameter("THREAD_SIZE_M_VALUE", is_tunable=True, search_space=[2, 4, 8])
        self.add_parameter("THREAD_SIZE_N_VALUE", is_tunable=True, search_space=[2, 4, 8])
        self.add_parameter("THREAD_SIZE_K_VALUE", is_tunable=True, search_space=[2, 4, 8])

    def check_parameters(self):
        M = self.get_parameter('GLOBAL_M_VALUE')
        K = self.get_parameter('GLOBAL_K_VALUE')
        N = self.get_parameter('GLOBAL_N_VALUE')
        BM = self.get_parameter('BLOCK_SIZE_M_VALUE')
        BK = self.get_parameter('BLOCK_SIZE_K_VALUE')
        BN = self.get_parameter('BLOCK_SIZE_N_VALUE')
        TM = self.get_parameter('THREAD_SIZE_M_VALUE')
        TK = self.get_parameter('THREAD_SIZE_K_VALUE')
        TN = self.get_parameter('THREAD_SIZE_N_VALUE')
        assert np.log2(BM) % 1 == 0
        assert np.log2(BK) % 1 == 0
        assert np.log2(BN) % 1 == 0
        assert np.log2(TM) % 1 == 0
        assert np.log2(TK) % 1 == 0
        assert np.log2(TN) % 1 == 0
        assert M % BM == 0
        assert K % BK == 0
        assert N % BN == 0
        assert BM % TM == 0
        assert BK % TK == 0
        assert BN % TN == 0
        assert M > BM
        assert K > BK
        assert N > BN
        assert BM > TM
        assert BK > TK
        assert BN > TN

    def set_ports_layout(self):
        BM = self.get_parameter('BLOCK_SIZE_M_VALUE')
        BK = self.get_parameter('BLOCK_SIZE_K_VALUE')
        BN = self.get_parameter('BLOCK_SIZE_N_VALUE')
        if self._stype == 'sdd':
            self.set_input_layout('A', {
                'mode': 'H' if self._compressed else 'HD',
                'block_size': [BM, BK],
            })
        elif self._stype == 'dsd':
            if self._transpose:
                self.set_input_layout('B', {
                    'mode': 'H' if self._compressed else 'HD',
                    'block_size': [BN, BK],
                })
            else:
                self.set_input_layout('B', {
                    'mode': 'V' if self._compressed else 'VD',
                    'block_size': [BK, BN],
                })
        elif self._stype == 'dds':
            self.set_output_layout('C', {
                'mode': 'X' if self._compressed else 'XD',
                'block_size': [BM, BN],
            })

    def blocks_per_grid(self) -> tuple[int]:
        M = self.get_parameter('GLOBAL_M_VALUE')
        N = self.get_parameter('GLOBAL_N_VALUE')
        BM = self.get_parameter('BLOCK_SIZE_M_VALUE')
        BN = self.get_parameter('BLOCK_SIZE_N_VALUE')
        return (N // BN, M // BM)

    def threads_per_block(self) -> tuple[int]:
        BM = self.get_parameter('BLOCK_SIZE_M_VALUE')
        BN = self.get_parameter('BLOCK_SIZE_N_VALUE')
        TM = self.get_parameter('THREAD_SIZE_M_VALUE')
        TN = self.get_parameter('THREAD_SIZE_N_VALUE')
        return (BN // TN, BM // TM)


class OpenAITemplateSparseMatMulKernel(MatMulKernelBase):

    def get_kernel_name(self) -> str:
        return f'openai_{super().get_kernel_name()}'

    def get_kernel_code(self) -> str:
        with open(os.path.join(TEMPLATE_DIR, f'openai_sparse_matmul_{self._stype}.cuh.j2')) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())

    def check_parameters(self):
        assert 32 < self.get_parameter('GLOBAL_M_VALUE')
        assert 64 < self.get_parameter('GLOBAL_K_VALUE')
        assert 32 < self.get_parameter('GLOBAL_N_VALUE')

    def set_ports_layout(self):
        BM = 32
        BK = 64
        BN = 32
        if self._stype == 'sdd':
            self.set_input_layout('A', {
                'mode': 'H' if self._compressed else 'HD',
                'block_size': [BM, BK],
            })
        elif self._stype == 'dsd':
            if self._transpose:
                self.set_input_layout('B', {
                    'mode': 'H' if self._compressed else 'HD',
                    'block_size': [BN, BK],
                })
            else:
                self.set_input_layout('B', {
                    'mode': 'V' if self._compressed else 'VD',
                    'block_size': [BK, BN],
                })
        elif self._stype == 'dds':
            self.set_output_layout('C', {
                'mode': 'X' if self._compressed else 'XD',
                'block_size': [BM, BN],
            })

    def blocks_per_grid(self) -> tuple[int]:
        if self._stype == 'dds':
            return (int(self.get_output('C').sparse()['nnz'][0]), )
        else:
            M = self.get_parameter('GLOBAL_M_VALUE')
            N = self.get_parameter('GLOBAL_N_VALUE')
            return (N // 32, M // 32)

    def threads_per_block(self) -> tuple[int]:
        return (256, )
