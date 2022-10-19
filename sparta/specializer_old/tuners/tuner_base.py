# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Dict, List, Generator


class TunerBase(abc.ABC):

    def __init__(self, search_space: Dict[str, List[int]] = None):
        self._search_space = search_space

    @abc.abstractmethod
    def _configs(self) -> Generator[Dict[str, int], None, None]:
        '''
        Generator that yields the next config to test
        '''
