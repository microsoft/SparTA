#include "torch/extension.h"

at::Tensor cusparse_linear_forward(
    torch::Tensor input,
    torch::Tensor row_index,
    torch::Tensor col_index,
    torch::Tensor values,
    std::vector<int> weight_shape);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &cusparse_linear_forward, "cuSparse sparse forward");
}