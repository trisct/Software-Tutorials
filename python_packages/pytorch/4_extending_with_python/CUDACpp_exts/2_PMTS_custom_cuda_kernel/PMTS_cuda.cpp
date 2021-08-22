// This implements the C++ interface and
// cuda kernels of a custom function that takes
// two tensors `a` and `b` as inputs, and returns
// two tensors `c` and `d` where
// c: sigmoid(tanh(a + b))
// d: sigmoid(tanh(a - b))
// this function will be called PMTS
// for PlusMinusTanhSigmoid


#include "PMTS_cuda.h"

// sanity check macros

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interfaces

std::vector<torch::Tensor> PMTS_forward(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    return PMTS_forward_cuda(a, b);
}

std::vector<torch::Tensor> PMTS_backward(torch::Tensor grad_pts, torch::Tensor grad_mts, torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(grad_pts);
    CHECK_INPUT(grad_mts);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    
    return PMTS_backward_cuda(grad_pts, grad_mts, a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &PMTS_forward, "PMTS forward (custom cuda kernel)");
  m.def("backward", &PMTS_backward, "PMTS backward (custom cuda kernel)");
}