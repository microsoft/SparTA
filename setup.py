# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

rootdir = os.path.dirname(os.path.realpath(__file__))

version = '1.0'
ext_modules = []
if torch.cuda.is_available():
    moe_ext = CUDAExtension(name='sparse_moe', sources=['csrc/moe_sparse_forward.cpp',\
                                                            'csrc/moe_sparse_forward_kernel.cu'],
                                extra_compile_args=['-std=c++14', '-O3', "-U__CUDA_NO_HALF_OPERATORS__",
                                                    "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF_CONVERSIONS__"])
    ext_modules.append(moe_ext)
    seqlen_dynamic_attention_ext = CUDAExtension(name='seqlen_dynamic_sparse_attention_cpp', sources=['csrc/seqlen_dynamic_sparse_attention_forward.cpp',
                                                            'csrc/seqlen_dynamic_sparse_attention_forward_kernel.cu'],
                                extra_compile_args=['-std=c++14', '-O3'])
    ext_modules.append(seqlen_dynamic_attention_ext)

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
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
    package_data={
        'sparta.specializer.kernels': ['templates/*.j2', 'look_up_tables/*.csv'],
        'sparta.tesa': ['templates/*.j2'],
    },
)
