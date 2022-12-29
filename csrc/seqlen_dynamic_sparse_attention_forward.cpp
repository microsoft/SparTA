
#include <vector>
#include "torch/extension.h"

at::Tensor seqlen_dynamic_sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor inter_result,
    torch::Tensor seqlens,
    int head_num
);
std::vector<at::Tensor> seqlen_dynamic_sparse_attention_backward(
    torch::Tensor grad,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
    );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &seqlen_dynamic_sparse_attention_forward, "our sparse attention forward");
    m.def("backward", &seqlen_dynamic_sparse_attention_backward, "our sparse attention backward");

}
