# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from setuptools import setup, find_packages

rootdir = os.path.dirname(os.path.realpath(__file__))

version = '1.0'

setup(
    name='SparTA',
    version=version,
    description='Deployment tool',
    author='MSRA',
    author_email='spartadev@microsoft.com',
    packages=find_packages(exclude=['test', 'test.*', 'examples', 'examples.*']),
    install_requires=[
        'jinja2',
        'pycuda',  # 'pip install pycuda' works for most cases
        'nni',
    ],
    include_package_data=True,
    package_data={
        'sparta.specializer.kernels': ['templates/*.j2'],
    },
)
