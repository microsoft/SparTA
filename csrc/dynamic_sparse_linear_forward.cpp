
#include <vector>
#include "torch/extension.h"

at::Tensor dynamic_sparse_linear_forward(
    torch::Tensor activation,
    torch::Tensor row_ptr,
    torch::Tensor col_indx,
    torch::Tensor val,
    torch::Tensor bias,
    int M, int K, int N, int block_h, int block_w
);

std::vector<at::Tensor> dynamic_sparse_linear_backward(
    torch::Tensor activation,
    torch::Tensor grad_a_row_ptr,
    torch::Tensor grad_a_col_index,
    torch::Tensor grad_a_val,
    torch::Tensor grad_c,
    int M, int K, int N, int block_h, int block_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &dynamic_sparse_linear_forward, "dynamic sparse linear forward");
    m.def("backward", &dynamic_sparse_linear_backward, "dynamic sparse linear backward");
}
