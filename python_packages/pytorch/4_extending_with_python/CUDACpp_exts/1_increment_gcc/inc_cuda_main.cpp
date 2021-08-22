#include "inc_cuda.h"

int main() {
    torch::Tensor a = torch::zeros({3, 2}).to(torch::kCUDA);
    auto b = increase_by_one(a);
    std::cout << b << std::endl;
}

