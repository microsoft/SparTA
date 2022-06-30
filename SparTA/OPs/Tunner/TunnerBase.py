# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Dict, List
import logging

from ..Emitter.EmitterBase import EmitterBase


class TunnerBase(abc.ABC):

    def __init__(self, emitter: EmitterBase, search_space: Dict[str, List[int]], logger: logging.Logger):
        self._emitter = emitter
        self._search_space = search_space
        self._logger = logger

    @abc.abstractmethod
    def tunning_kernel_cfg(self, *args, **kwargs):
        """
        Measure the latency of a specific kernel of the template.
        """
