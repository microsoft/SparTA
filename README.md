# SparTA Getting Started

> *This repo is current an **alpha** version and under active development. We're trying to make it stable and user-friendly, but it is not surprising to meet problems in current phase. Please open issue or contact the authors when you need help.*

`SparTA` is an end-to-end system to harvest the speeding up gain from the model sparsity.

## Installation 
SparTA depends on user's local CUDA environments. Here are some requirements
- [PyTorch](https://pytorch.org/)

User could install through `pip` command as below (*The PyPI install path is coming soon*)
```bash
pip install git+https://github.com/microsoft/SparTA.git
```
or
```bash
git clone git@github.com:microsoft/SparTA.git
pip install SparTA
```

Please make sure that the CUDA version matches the version used to compile PyTorch binaries.
If cuda and nvcc version issues met, the following commands may be helpful to verify the environments. 

```python
import os
import torch
import pycuda.driver

if torch.cuda.is_available():
    os.system('nvcc --version')
    print(torch.version.cuda)
    print(pycuda.driver.get_version())
    print(pycuda.driver.get_driver_version())
```

## Usage

### Tune a sparse operator

```python
import torch
import sparta

M, N, K = 1024, 1024, 1024
SPARSITY = 0.8
BLOCK = (8, 8)
has_bias = True

A = torch.rand((M,K), dtype=torch.float32).cuda()
B = torch.rand((N,K), dtype=torch.float32).cuda()
bias = torch.rand((N,),dtype=torch.float32).cuda()
# generate and apply mask
B_mask = sparta.testing.block_mask(B.shape, block=BLOCK, sparsity=SPARSITY).cuda()
B = torch.mul(B, B_mask)
# dense operator
linear = torch.nn.Linear(K, N, bias=has_bias).cuda()
linear.load_state_dict(dict(weight=B, bias=bias) if has_bias else dict(weight=B) )
# sparse operator
splinear = sparta.nn.SparseLinear(linear, weight_mask=B_mask)
best_cfg = sparta.nn.tune(splinear, sample_inputs=[A], max_trials=10, algo='rand')
torch.testing.assert_close(splinear(A), linear(A))
```

## Citing SparTA
If SparTA is helpful in your projects, please cite our paper as below
```
@inproceedings {SparTA2022,
    author = {Ningxin Zheng and Bin Lin and Quanlu Zhang and Lingxiao Ma and Yuqing Yang and Fan Yang and Yang Wang and Mao Yang and Lidong Zhou},
    title = {SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute},
    booktitle = {16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)},
    year = {2022},
    isbn = {978-1-939133-28-1},
    address = {Carlsbad, CA},
    pages = {213--232},
    url = {https://www.usenix.org/conference/osdi22/presentation/zheng-ningxin},
    publisher = {USENIX Association},
    month = jul,
}
```

## Contributor License Agreement

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.