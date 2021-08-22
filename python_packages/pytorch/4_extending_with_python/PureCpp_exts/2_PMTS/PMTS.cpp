#include <torch/torch.h>
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

int main() {
  torch::Tensor a = torch::rand({3, 2});
  torch::Tensor b = torch::rand({3, 2});
  std::vector<torch::Tensor> c = PMTS_forward(a, b);
  std::cout << c[0] << std::endl;
  std::cout << c[1] << std::endl;
}
