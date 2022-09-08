

from ctypes import Union
from dataclasses import dataclass
from typing import Optional, Any
import uuid
from hyperopt import hp
from hyperopt.pyll.stochastic import sample

@dataclass
class TunableItemCfg:
    algo: str # currently only support ['choice']
    args: list
    
    def __post_init__(self):
        assert self.algo in __hp_algos__
        if self.algo == 'choice':
            assert isinstance(self.args, list)

__hp_algos__ = {
    'choice': hp.choice,
    'pchoice': hp.pchoice,
}


class Tunable:
    '''We're defining an easy definition to write a config
    '''

    def __init__(self, search_space_cfg: TunableItemCfg, name: str = None, backend: str = 'hyperopt') -> None:
        self.search_space_cfg = search_space_cfg
        self.search_space = None
        self.backend = backend
        self.name = name or str(uuid.uuid4())[:8]
        if self.search_space_cfg:
            self.parse_config()
        
    def parse_config(self):
        def _recursive(obj, n: str):
            if isinstance(obj, TunableItemCfg):
                if self.backend == 'hyperopt':
                    return __hp_algos__[obj.algo](n, [_recursive(x, f'{n}::{i}') for i,x in enumerate(obj.args)])                        
            if isinstance(obj, list):
                return [_recursive(x, f'{n}::{i}') for i,x in enumerate(obj)]
            if isinstance(obj, dict):
                return {k: _recursive(v, f'{n}::{k}') for k,v in obj.items()}
            return obj
        self.search_space = _recursive(self.search_space_cfg, self.name)

    def sample(self):
        assert self.search_space is not None
        return sample(self.search_space)


if __name__ == '__main__':
    search_space_cfg = TunableItemCfg('choice', [
        {'implement': 'openai'},
        {
            'implement': 'sparta',
            'config':{
                'BLOCK_SIZE_M_VALUE': TunableItemCfg('choice', [32, 64]),
                'BLOCK_SIZE_K_VALUE': TunableItemCfg('choice', [32, 64]),
                'BLOCK_SIZE_N_VALUE': TunableItemCfg('choice', [32, 64]),
                'THREAD_SIZE_M_VALUE': TunableItemCfg('choice', [4]),
                'THREAD_SIZE_K_VALUE': TunableItemCfg('choice', [4]),
                'THREAD_SIZE_N_VALUE': TunableItemCfg('choice', [4]),
            }
        },
    ])

    t = Tunable(search_space_cfg, name='linear')
    # print(t.search_space)
    for i in range(10):
        print(t.sample())    