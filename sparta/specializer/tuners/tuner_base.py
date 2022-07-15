# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Dict, List

from sparta.specializer.factories import factory_base


class TunerBase(abc.ABC):

    def __init__(self, factory: factory_base.FactoryBase, search_space: Dict[str, List[int]]):
        self._factory = factory
        self._search_space = search_space

    @abc.abstractmethod
    def tunning_kernel_cfg(self, *args, **kwargs):
        """
        Measure the latency of a specific kernel of the template.
        """
