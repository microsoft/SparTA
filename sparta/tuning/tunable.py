# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Optional, Any, Union, Dict, List


@dataclass
class TunableItemCfg:
    """TunableItemCfg is used to describe the search space

    Examples:

        .. code-block:: python

            cfg = TunableItemCfg('choice', _is_nested=True, _value={
                'openai': {},
                'sparta': {
                    'BM': TunableItemCfg('choice', [32, 64]),
                    'BN': TunableItemCfg('choice', [8, 16]),
                }
            })
            nni_space = {
                'test': {
                    '_type': 'choice', '_value': [
                        {'_name': 'openai'},
                        {
                            '_name': 'sparta',
                            'BM': {'_type': 'choice', '_value': [32, 64]},
                            'BN': {'_type': 'choice', '_value': [8, 16]},
                        }
                    ]
                }
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
            assert isinstance(self._value, dict)
            for ss_name, ss_params in self._value.items():
                assert isinstance(ss_params, dict)
                for p_name, p_cfg in ss_params.items():
                    assert isinstance(p_cfg, TunableItemCfg)
        else:
            assert isinstance(self._value, list)

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
