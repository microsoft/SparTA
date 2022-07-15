# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json

from SparTA.Specializer.Factories.FactoryBase import FactoryBase
from SparTA.Specializer.Factories.TemplateBasedFactory.TemplateBasedFactory import TemplateBasedFactory


OPERATOR_CONFIG_DIR = os.path.join('SparTA', 'Specializer', 'Configs')


class Specializer(object):

    def get_factory(self, op_config_file: str) -> 'FactoryBase':
        with open(os.path.join(OPERATOR_CONFIG_DIR, op_config_file)) as f:
            op_config = json.loads(f.read())
        if op_config['factory_type'] == 'template':
            return TemplateBasedFactory(op_config)

