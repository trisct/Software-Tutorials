// This implements the C++ interface of a 
// custom function that takes
// two tensors `a` and `b` as inputs, and returns
// two tensors `c` and `d` where
// c: sigmoid(tanh(a + b))
// d: sigmoid(tanh(a - b))
// this function will be called PMTS
// for PlusMinusTanhSigmoid

#include <torch/extension.h>
#include <iostream>

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
    return 1 - z.tanh().pow(2);
}

// sigmoid'(z) = (1 - sigmoid(z)) * sigmoid(z)
torch::Tensor d_sigmoid(torch::Tensor z) {
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}

std::vector<torch::Tensor> PMTS_forward(torch::Tensor a, torch::Tensor b) {
    auto PTS = (a + b).tanh().sigmoid();
    auto MTS = (a - b).tanh().sigmoid();
    return {PTS, MTS};
}

std::vector<torch::Tensor> PMTS_backward(torch::Tensor grad_PTS, torch::Tensor grad_MTS, torch::Tensor a, torch::Tensor b) {
    auto grad_PTS_a = grad_PTS * d_sigmoid((a + b).tanh()) * d_tanh(a + b);
    auto grad_PTS_b = grad_PTS * d_sigmoid((a + b).tanh()) * d_tanh(a + b);
    auto grad_MTS_a = grad_MTS * d_sigmoid((a - b).tanh()) * d_tanh(a - b);
    auto grad_MTS_b = grad_MTS * d_sigmoid((a - b).tanh()) * d_tanh(a - b) * -1;
    
    auto grad_a = grad_PTS_a + grad_MTS_a;
    auto grad_b = grad_PTS_b + grad_MTS_b;
    
    return {grad_a, grad_b};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &PMTS_forward, "PMTS forward");
    m.def("backward", &PMTS_backward, "PMTS backward");
}