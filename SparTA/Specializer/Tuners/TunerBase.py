# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Dict, List

from SparTA.Specializer.Factories.FactoryBase import FactoryBase


class TunerBase(abc.ABC):

    def __init__(self, factory: FactoryBase, search_space: Dict[str, List[int]]):
        self._factory = factory
        self._search_space = search_space

    @abc.abstractmethod
    def tunning_kernel_cfg(self, *args, **kwargs):
        """
        Measure the latency of a specific kernel of the template.
        """
