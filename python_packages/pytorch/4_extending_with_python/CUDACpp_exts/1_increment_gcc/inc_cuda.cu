#include "inc_cuda.h"


template <typename scalar_t>
__global__ void increase_by_one_cuda_kernel(scalar_t* __restrict__ a, size_t linear_size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // using linear configuration here
    
    if (idx < linear_size)
        a[idx] += 1;

    __syncthreads();
}

torch::Tensor increase_by_one_cuda(torch::Tensor a) {
    auto n_dims = a.dim();
    auto linear_size = a.size(0);

    for (int i = 1; i < n_dims; i++)
        linear_size *= a.size(i);

    
    const int threads = 1024;
    const int blocks = (linear_size + threads - 1) / threads;

    auto ans = torch::zeros_like(a);

    AT_DISPATCH_FLOATING_TYPES(a.type(), "increase by one", ([&] {
        increase_by_one_cuda_kernel<scalar_t><<<blocks, threads>>>(
            a.data<scalar_t>(),
            linear_size);
    }));

    return ans;
}
