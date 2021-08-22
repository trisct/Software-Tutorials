#include "PMTS_cuda.h"

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid_custom(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid_custom(scalar_t z) {
  auto s = 1.0 / (1.0 + exp(-z));
  return (1 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t tanh_custom(scalar_t z) {
  auto ez = exp(z);
  auto emz = exp(-z);
  return (ez - emz) / (ez + emz);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh_custom(scalar_t z) {
  auto ez = exp(z);
  auto emz = exp(-z);
  auto tanhz = (ez - emz) / (ez + emz);
  return 1 - tanhz * tanhz;
}

template <typename scalar_t>
__global__ void PMTS_forward_cuda_kernel(
        scalar_t* __restrict__ input_a,
        scalar_t* __restrict__ input_b,
        scalar_t* __restrict__ pts,
        scalar_t* __restrict__ mts,
        size_t input_linear_size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // using linear configuration here
    
    if (idx < input_linear_size) {
        pts[idx] = sigmoid_custom(tanh_custom(input_a[idx] + input_b[idx]));
        mts[idx] = sigmoid_custom(tanh_custom(input_a[idx] - input_b[idx]));
    }

    __syncthreads();
}

template <typename scalar_t>
__global__ void PMTS_backward_cuda_kernel(
        scalar_t* __restrict__ grad_pts,
        scalar_t* __restrict__ grad_mts,
        scalar_t* __restrict__ a,
        scalar_t* __restrict__ b,
        scalar_t* __restrict__ grad_a,
        scalar_t* __restrict__ grad_b,
        size_t input_linear_size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // using linear configuration here
    
    if (idx < input_linear_size) {
        scalar_t plus_idx = a[idx] + b[idx];
        scalar_t minus_idx = a[idx] - b[idx];

        scalar_t grad_pts_idx =
            grad_pts[idx] *
            d_sigmoid_custom(tanh_custom(plus_idx)) *
            d_tanh_custom(plus_idx);
        scalar_t grad_mts_a_idx =
            grad_mts[idx] *
            d_sigmoid_custom(tanh_custom(minus_idx)) *
            d_tanh_custom(minus_idx);
        scalar_t grad_mts_b_idx =
            grad_mts[idx] *
            d_sigmoid_custom(tanh_custom(minus_idx)) *
            d_tanh_custom(minus_idx) *
            -1;
        
        scalar_t grad_a_idx = grad_pts_idx + grad_mts_a_idx;
        scalar_t grad_b_idx = grad_pts_idx + grad_mts_b_idx;
        
        grad_a[idx] = grad_a_idx;
        grad_b[idx] = grad_b_idx;
    }
    
    __syncthreads();
}

std::vector<torch::Tensor> PMTS_forward_cuda(
        torch::Tensor a,
        torch::Tensor b) {

    auto n_dims = a.dim();
    auto linear_size = a.size(0);

    for (int i = 1; i < n_dims; i++) {
        linear_size *= a.size(i);
    }

    
    const int threads = 1024;
    const int blocks = (linear_size + threads - 1) / threads;

    auto pts = torch::zeros_like(a);
    auto mts = torch::zeros_like(a);

    AT_DISPATCH_FLOATING_TYPES(a.type(), "PMTS_forward_cuda_name", ([&] {
        PMTS_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            a.data<scalar_t>(),
            b.data<scalar_t>(),
            pts.data<scalar_t>(),
            mts.data<scalar_t>(),
            linear_size);
    }));

    return {pts, mts};
}

std::vector<torch::Tensor> PMTS_backward_cuda(
        torch::Tensor grad_pts,
        torch::Tensor grad_mts,
        torch::Tensor a,
        torch::Tensor b) {

    auto n_dims = a.dim();
    auto linear_size = a.size(0);

    for (int i = 1; i < n_dims; i++) {
        linear_size *= a.size(i);
    }

    const int threads = 1024;
    const int blocks = (linear_size + threads - 1) / threads;

    auto grad_a = torch::zeros_like(a);
    auto grad_b = torch::zeros_like(b);

    AT_DISPATCH_FLOATING_TYPES(a.type(), "PMTS_backward_cuda", ([&] {
        PMTS_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            grad_pts.data<scalar_t>(),
            grad_mts.data<scalar_t>(),
            a.data<scalar_t>(),
            b.data<scalar_t>(),
            grad_a.data<scalar_t>(),
            grad_b.data<scalar_t>(),
            linear_size);
    }));

    return {grad_a, grad_b};
}