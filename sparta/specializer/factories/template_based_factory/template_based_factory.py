# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import jinja2

from sparta.specializer import factories


KERNEL_TEMPLATE_DIR = os.path.join('sparta', 'specializer', 'factories', 'template_based_factory', 'kernels')


class TemplateBasedFactory(factories.FactoryBase):

    def __init__(self, op_config: dict):
        super().__init__(op_config)
        with open(os.path.join(KERNEL_TEMPLATE_DIR, f'{self.name}.cuh.j2')) as f:
            self._template = f.read()

    def get_kernel_code(self, **kwargs) -> str:
        return jinja2.Template(self._template).render(kwargs)
