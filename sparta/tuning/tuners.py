# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import sys
import random
import logging
from typing import Any, Dict, Callable, Iterator

import numpy as np

from sparta.tuning.tunable import TunableItemCfg


_logger = logging.Logger(__name__)
_handler = logging.StreamHandler()
_logger.addHandler(_handler)


class Tuner(object):

    def __init__(
        self,
        search_space: Dict[Any, TunableItemCfg],
        eval_func: Callable[[int, Dict[Any, Any]], float],
        max_trials: int = sys.maxsize,
    ):
        self._search_space = search_space
        self._eval_func = eval_func
        space_shape = [len(param_space._value) for param_space in search_space.values()]
        space_size = int(np.prod(space_shape))
        self._max_trials = min(max_trials, space_size)
        self.best_result = np.inf
        self.best_config = None

    @abc.abstractmethod
    def next_config(self) -> Iterator[Dict[str, Any]]:
        """Yields the next config."""

    def tune(self):
        for i, config in zip(range(self._max_trials), self.next_config()):
            result = self._eval_func(i, config)
            if result < self.best_result:
                self.best_result = result
                self.best_config = config


class RandomSearchTuner(Tuner):

    def next_config(self):
        while True:
            yield {
                param_name: random.choice(param_space._value)
                for param_name, param_space in self._search_space.items()
            }


class GridSearchTuner(Tuner):

    def next_config(self):
        if len(self._search_space) == 0:
            yield {}
        else:
            param_names = []
            param_idxs = []
            param_space_sizes = []
            for param_name, param_space in self._search_space.items():
                param_names.append(param_name)
                param_space_sizes.append(len(param_space._value))
                param_idxs.append(0)
            while param_idxs[0] < param_space_sizes[0]:
                yield {
                    param_name: self._search_space[param_name]._value[param_idx]
                    for param_idx, param_name in zip(param_idxs, param_names)
                }
                k = len(param_idxs) - 1
                param_idxs[k] += 1
                while param_idxs[k] == param_space_sizes[k] and k > 0:
                    param_idxs[k - 1] += 1
                    param_idxs[k] = 0
                    k -= 1
