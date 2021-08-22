#ifndef __PMTS_CUDA_H__
#define __PMTS_CUDA_H__

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

torch::Tensor increase_by_one(torch::Tensor a);
torch::Tensor increase_by_one_cuda(torch::Tensor a);

#endif
