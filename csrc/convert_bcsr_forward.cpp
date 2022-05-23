#include "torch/extension.h"
#include <vector>

std::vector<at::Tensor> convert_bcsr_forward(
    torch::Tensor sparse_pattern,
    torch::Tensor dense_values,
    int block_h, 
    int block_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &convert_bcsr_forward, "Convert block csr format");
}