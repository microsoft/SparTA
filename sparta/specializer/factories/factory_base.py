# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Callable

import torch
import numpy as np


class FactoryBase(abc.ABC):

    def __init__(self, op_config: dict):
        self.name = op_config['name']
        self.dims = op_config['dims']
        self.inputs = op_config['inputs']
        self.outputs = op_config['outputs']
        self.tiles = op_config['tiles']

    @abc.abstractclassmethod
    def get_test_func(self, shape_config: dict, mask: dict[str, 'np.ndarray'] = None) -> Callable:
        '''
        Get a callable test function
        '''

    @abc.abstractclassmethod
    def get_module(self, shape_config: dict, mask: dict[str, 'np.ndarray'] = None) -> 'torch.nn.Module':
        '''
        Get a pytorch module
        '''
