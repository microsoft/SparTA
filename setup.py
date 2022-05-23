
import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

rootdir = os.path.dirname(os.path.realpath(__file__))

version = "0.0.1"

ext_modules = []

if torch.cuda.is_available():
    # Add the extral sparse kernel module
    # cusparse need cuda version higher than 11.1
    cuda_version = torch.version.cuda
    if cuda_version > '11.1':
        cusparse_ext = CUDAExtension(name='cusparse_linear', sources=[
                                'csrc/cusparse_linear_forward.cpp', 'csrc/cusparse_linear_forward_kernel.cu'],
                                extra_compile_args=['-std=c++14', '-lcusparse', '-O3'])
        ext_modules.append(cusparse_ext)
    # the bcsr convert kernel
    bcsr_ext = CUDAExtension(name='convert_bcsr', sources=['csrc/convert_bcsr_forward.cpp',
                                                           'csrc/convert_bcsr_forward_kernel.cu'],
                                extra_compile_args=['-std=c++14', '-O3'])
    ext_modules.append(bcsr_ext)
    dynamic_attention_ext = CUDAExtension(name='dynamic_sparse_attention', sources=['csrc/dynamic_sparse_attention_forward.cpp',
                                                           'csrc/dynamic_sparse_attention_forward_kernel.cu'],
                                extra_compile_args=['-std=c++14', '-O3'])
    ext_modules.append(dynamic_attention_ext)
    dynamic_linear_ext = CUDAExtension(name='dynamic_sparse_linear', sources=['csrc/dynamic_sparse_linear_forward.cpp',
                                                            'csrc/dynamic_sparse_linear_forward_kernel.cu'],
                                extra_compile_args=['-std=c++14', '-O3'])
    ext_modules.append(dynamic_linear_ext)
    # cusparse_ext = CUDAExtension(name='our_sparse_attention', sources=[
    #                             'csrc/sparse_attention.cpp', 'csrc/sparse_attention_kernel.cu'],
    #                             extra_compile_args=['-std=c++14', '-O3'])
    # ext_modules.append(cusparse_ext)


setup(
    name='SparTA',
    version=version,
    description='Deployment tool',
    author='MSRA',
    author_email="Ningxin.Zheng@microsoft.com",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
