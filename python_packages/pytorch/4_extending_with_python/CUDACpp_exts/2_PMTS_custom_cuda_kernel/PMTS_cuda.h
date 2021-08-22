#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


#ifndef __PMTS_CUDA_H__
#define __PMTS_CUDA_H__

// CUDA interface declarations, defined in the corresponding .cu file

std::vector<torch::Tensor> PMTS_forward_cuda(torch::Tensor a, torch::Tensor b);
std::vector<torch::Tensor> PMTS_backward_cuda(torch::Tensor grad_pts, torch::Tensor grad_mts, torch::Tensor a, torch::Tensor b);

// C++ interfaces

std::vector<torch::Tensor> PMTS_forward(torch::Tensor a, torch::Tensor b);
std::vector<torch::Tensor> PMTS_backward(torch::Tensor grad_pts, torch::Tensor grad_mts, torch::Tensor a, torch::Tensor b);

#endif