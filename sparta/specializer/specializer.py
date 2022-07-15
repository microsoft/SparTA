# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json

from sparta.specializer import factories


OPERATOR_CONFIG_DIR = os.path.join('sparta', 'specializer', 'configs')


class Specializer(object):

    def get_factory(self, op_config_file: str) -> 'factories.FactoryBase':
        with open(os.path.join(OPERATOR_CONFIG_DIR, op_config_file)) as f:
            op_config = json.loads(f.read())
        if op_config['factory_type'] == 'template':
            return factories.TemplateBasedFactory(op_config)
