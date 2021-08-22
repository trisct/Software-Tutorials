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
    auto minus = (a - b).;
    auto 
}