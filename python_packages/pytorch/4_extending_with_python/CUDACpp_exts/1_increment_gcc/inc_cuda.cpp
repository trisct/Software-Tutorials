#include "inc_cuda.h"

// sanity check macros

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interfaces

torch::Tensor increase_by_one(torch::Tensor a) {
    CHECK_INPUT(a);
    return increase_by_one_cuda(a);
}


