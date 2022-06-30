# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import json
import copy
import logging
from sys import stderr
from typing import Dict, List

import numpy as np

from ...Common.Utils import cuda_detect, call_shell
from .EmitterBase import EmitterBase
from ..Template import BlockSparseLinear

class BlockSparseEmitter(EmitterBase):
    def __init__(self, M: int, K: int, N: int, tmp_dir: str):
        temlate_path = os.path.join(os.path.split(__file__)[0], '../Template')
        cfg_path = os.path.join(temlate_path, 'BlockSparseLinear.json')
        assert os.path.exists(cfg_path)
        with open(cfg_path, 'r') as f:
            self.template_cfg = json.load(f)

        code_template = BlockSparseLinear.block_sparse_linear_function_template
        self.function_body, self.function_call = self.parse_code_template(code_template)
        self.header = BlockSparseLinear.block_sparse_linear_header_template
        self.main_body = BlockSparseLinear.block_sparse_linear_test_template

        self.tmp_dir = tmp_dir # directory used to compile and test the program
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.inputs_dir = os.path.join(self.tmp_dir, 'inputs')
        if not os.path.exists(self.inputs_dir):
            os.makedirs(self.inputs_dir)

        self.shape = {"GLOBAL_M_VALUE": M, "GLOBAL_K_VALUE": K, "GLOBAL_N_VALUE": N}
        for key, value in self.shape.items():
            self.header =  self.header.replace(key, str(value))
        for key, value in self.template_cfg['inputs'].items():
            self.main_body =  self.main_body.replace(f'INPUT_FILE_PATH_{key.upper()}', f'"{value}"')

        self.gpu_devices = cuda_detect()
        assert len(self.gpu_devices) > 0
        self.gpu_code = self.gpu_devices[0][1]

    def parse_code_template(self, code_template: str):
        function_body = code_template
        function_call = code_template.split('{')[0]
        return function_body, function_call

    def emit_function_body(self, stringstream):
        stringstream.write(self.function_body + '\n')

    def emit_function_call(self, stringstream):
        stringstream.write(self.funciton_call + '\n')

    def emit_dependency(self, stringstream, kw_args: Dict[str, int] = {}):
        header = copy.deepcopy(self.header)        
        for key, value in kw_args.items():
            header = header.replace(key, str(value))
        stringstream.write(header + '\n')

    def emit_test_main(self, stringstream):
        stringstream.write(self.main_body)

    def write_input(self, input_name: str, input_matrix: np.array, format: str = '%.2f'):
        assert len(input_matrix.shape) == 1
        with open(os.path.join(self.inputs_dir, self.template_cfg['inputs'][input_name]), 'w') as f:
            f.write(' '.join(map(lambda x: format % x, input_matrix)))

    def block_cover(self, W: np.array, cfg: Dict[str, int]):
        block_size_k = cfg['BLOCK_SIZE_K_VALUE']
        block_size_n = cfg['BLOCK_SIZE_N_VALUE']
        block_num_k = self.shape['GLOBAL_K_VALUE'] // block_size_k
        block_num_n = self.shape['GLOBAL_N_VALUE'] // block_size_n
        W_row = np.zeros(block_num_k + 1)
        W_col = np.array([])
        W_val = np.array([])
        for block_i in range(block_num_k):
            block_start_k = block_i * block_size_k
            block_end_k = block_start_k + block_size_k
            for block_j in range(block_num_n):
                block_start_n = block_j * block_size_n
                block_end_n = block_start_n + block_size_n
                block = W[block_start_k:block_end_k, block_start_n:block_end_n]
                if np.abs(block).sum() > 0:
                    W_col = np.append(W_col, block_j)
                    W_val = np.concatenate([W_val, block.flatten()])
            W_row[block_i + 1] = len(W_col)
        self.write_input('W_row', W_row.flatten(), format='%d')
        self.write_input('W_col', W_col.flatten(), format='%d')
        self.write_input('W_val', W_val.flatten())

    def load_inputs(self, A: np.array, W: np.array, bias: np.array, cfg: Dict[str, int]):
        assert A.shape == (self.shape['GLOBAL_M_VALUE'], self.shape['GLOBAL_K_VALUE'])
        self.write_input('A', A.flatten())
        assert W.shape == (self.shape['GLOBAL_K_VALUE'], self.shape['GLOBAL_N_VALUE'])
        self.block_cover(W, cfg)
        assert bias.shape == (self.shape['GLOBAL_N_VALUE'], )
        self.write_input('bias', bias.flatten())

    def measure_trail_latency(self, A: np.array, W: np.array, bias: np.array, cfg: Dict[str, int], logger: logging.Logger = None) -> float:
        kernel_f_path = os.path.join(self.tmp_dir, 'block_sparse_kernel.cu')
        exec_path = os.path.join(self.tmp_dir, 'block_sparse')
        log_path = os.path.join(self.tmp_dir, 'block_sparse_run.log')
        kernel_f = open(kernel_f_path, 'w')
        self.emit_dependency(kernel_f, cfg)
        self.emit_function_body(kernel_f)
        self.emit_test_main(kernel_f)
        kernel_f.close()

        self.load_inputs(A, W, bias, cfg)
        # import pdb; pdb.set_trace()

        try:
            call_shell(f"nvcc -gencode arch=compute_{self.gpu_code},code=sm_{self.gpu_code} {kernel_f_path} -o {exec_path}", logger)
            call_shell(f"{exec_path} > {log_path}", logger)
            with open(log_path, 'r') as f:
                latency = float(f.readline())
        except Exception as e:
            latency = float('inf')
        return latency
