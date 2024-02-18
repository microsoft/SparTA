# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from setuptools import setup, find_packages

import jinja2
import torch
from torch.utils.cpp_extension import BuildExtension


version = '1.0'
ext_modules = []
if torch.cuda.is_available():
    from torch.utils.cpp_extension import CUDAExtension

    major, minor = torch.cuda.get_device_capability()
    with open(os.path.join('csrc', 'moe_sparse_forward_kernel.cu.j2')) as f:
        moe_template = f.read()
    moe_kernel = jinja2.Template(moe_template).render({'FP16': major >= 7})
    os.makedirs(os.path.join('csrc', 'build'), exist_ok=True)
    with open(os.path.join('csrc', 'build', 'moe_sparse_forward_kernel.cu'), 'w') as f:
        f.write(moe_kernel)

    moe_ext = CUDAExtension(
        name='sparta.sp_moe_ops',
        sources=[
            os.path.join('csrc', 'moe_sparse_forward.cpp'),
            os.path.join('csrc', 'build', 'moe_sparse_forward_kernel.cu'),
        ],
        extra_compile_args=[
            '-std=c++17',
            '-O3',
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
        ],
    )
    ext_modules.append(moe_ext)

    seqlen_dynamic_attention_ext = CUDAExtension(
        name='sparta.sp_attn_ops',
        sources=[
            os.path.join('csrc', 'seqlen_dynamic_sparse_attention_forward.cpp'),
            os.path.join('csrc', 'seqlen_dynamic_sparse_attention_forward_kernel.cu'),
        ],
        extra_compile_args=['-std=c++17', '-O3'],
    )
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
    cmdclass={'build_ext': BuildExtension},
    include_package_data=True,
    package_data={
        'sparta.kernels.templates': ['*.j2'],
        'sparta.kernels.look_up_tables': ['*.csv'],
        'sparta.tesa.templates': ['*.j2'],
    },
)
