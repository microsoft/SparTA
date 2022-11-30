# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Optional, Any, Union, Dict, List

from sparta.common.utils import check_type, get_uname


@dataclass
class TunableItemCfg:
    """TunableItemCfg is used to describe the search space
    Examples:

        .. code-block:: python

            cfg = TunableItemCfg('choice', _is_nested=True, _value={
                'openai': {},
                'sparta': {
                    'BM': TunableItemCfg('choice', [32,64]),
                    'BN': TunableItemCfg('choice', [8,16]),
                }
            })
            nni_space = {
                'test': {'_type':'choice', '_value': [
                    {'_name': 'openai'},
                    {
                        '_name': 'sparta',
                        'BM': {'_type': 'choice', '_value': [32,64]},
                        'BN': {'_type': 'choice', '_value': [8,16]},
                    }]}
            }

            # converted to a `NNI` search space (See more in https://nni.readthedocs.io/en/stable/hpo/search_space.html)
            assert search_space_cfg.to_nni_search_space() == nni_space    

    Args:
        _type (str): paramter type, allowed one of `('choice')`. 
        _value (Dict | List): options for paramter.
        _is_nested (bool): whether this space is nested (default: False). If True, the `_value` should be `Dict[str, Dict[TunableItemCfg]]`
    """
    _type: str  # currently only support ['choice']
    _value: Union[Dict, List]
    _is_nested: Optional[bool] = False

    def __post_init__(self):
        assert self._type in ['choice']
        if self._is_nested:
            check_type(self._value, dict)
            for ss_name, ss_params in self._value.items():
                check_type(ss_params, dict)
                for p_name, p_cfg in ss_params.items():
                    check_type(p_cfg, TunableItemCfg)
        else:
            check_type(self._value, list)

    def to_nni_search_space(self):
        """convert to nni search space"""
        if not self._is_nested:
            return {'_type': self._type, '_value': self._value}
        # self._value Dict[str, Dict[TunableItemCfg]]
        subspaces = []
        for ss_name, ss_dic in self._value.items():
            dic = {'_name': ss_name}
            for ss_item, ss_item_cfg in ss_dic.items():
                dic[ss_item] = ss_item_cfg.to_nni_search_space()
            subspaces.append(dic)
        return {'_type': self._type, '_value': subspaces}

    def includes(self, value: Any):
        if self._is_nested:
            return False
        else:
            return value in self._value


class Tunable:
    """The wrapper of NNI tuners that supports nested choice search space.
    """

    @staticmethod
    def supported_tuners():
        from nni.algorithms.hpo.gridsearch_tuner import GridSearchTuner
        from nni.algorithms.hpo.random_tuner import RandomTuner
        from nni.algorithms.hpo.tpe_tuner import TpeTuner
        from nni.algorithms.hpo.evolution_tuner import EvolutionTuner
        return {
            'grid': GridSearchTuner,
            'rand': RandomTuner,
            'tpe': TpeTuner,
            'evolution': EvolutionTuner,
        }

    @staticmethod        
    def create_tuner(algo: str, search_space_cfg: Dict[str, TunableItemCfg], tuner_kw: Dict = None):
        """create NNI Tuner

        Args:
            algo (str): tuning algorithm, allowed algo values and their corresponding tuners are:
            
                ========= ===================================================
                algo      tuner
                ========= ===================================================
                grid      nni.algorithms.hpo.gridsearch_tuner.GridSearchTuner
                rand      nni.algorithms.hpo.random_tuner.RandomTuner
                tpe       nni.algorithms.hpo.tpe_tuner.TpeTuner
                evolution nni.algorithms.hpo.evolution_tuner.EvolutionTuner
                ========= ===================================================
            
            search_space_cfg (TunableItemCfg): search space config
            tuner_kw (Dict): parameters passed to NNI tuner
        """
        supported_tuners = Tunable.supported_tuners()
        assert algo in supported_tuners, f'{algo} is not supported'
        tuner_kw = tuner_kw or {}
        tuner = supported_tuners[algo](**tuner_kw)
        tuner.update_search_space({k: v.to_nni_search_space() if v else {} for k, v in search_space_cfg.items()})
        return tuner
