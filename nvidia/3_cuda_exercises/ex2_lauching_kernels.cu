/// @brief on the purpose of this exercise:
// 1. understand how to define kernels
// 2. understand how to lauch kernels

/// @brief on what you can learn in this exercise:
// 1. kernel definition
// 2. kernel lauch configuration
// 3. blocks, threads and thread hierarchy

/// @brief on the pipeline of this exercise:
//  1. define a cuda kernel for vector addition
//  2. launch the kernel with specific configurations
//  3. __device__, __host__, and __global__ function execution space specifiers


/// @note on the function execution space specifiers
//  1. __global__
//      The __global__ specifier declares a function as being kernel, it is
//      - executed on the device
//      - callable from the host
//      - callable from the device
//      - must have void return type
//      - cannot be a member of a class
//      - needs a n execution configuration
//      - is asynchronuous
//  2. __device__
//      The __device__ speficier declares a function as
//      - executed on the device
//      - callable from the device only
//  3. __host__
//      - executed on the host
//      - callable from the host only
//  __global__ cannot be used together with __host__ or __device__
//  __device__ and __host__ can be used together, but two versions (for host and device) will be compiled.
//      Moreover, if you want two compilations to use different code paths, use the __CUDA_ARCH__ macro
 

#include "utils.h"

__global__ void vector_add_threadidx_x_only_kernel(float* A, float* B, float* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

__global__ void vector_add_blockidx_x_only_kernel(float* A, float* B, float* C) {
    int i = blockIdx.x;
    C[i] = A[i] + B[i];
}


int main() {
    int N = 10;
    size_t n_bytes = N * sizeof(float);

    // initialize host data
    float* h_A;
    float* h_B;
    float* h_C;
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    modulo_affine_init(h_A, N, 5, 1, 9);
    modulo_affine_init(h_B, N, 7, 2, 9);
    modulo_affine_init(h_C, N, 0, 0, 1);

    printf("initialization:\n");
    check_memory(h_A, N, "h_A");
    check_memory(h_B, N, "h_B");
    check_memory(h_C, N, "h_C");

    // device data
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, n_bytes);
    cudaMalloc(&d_B, n_bytes);
    cudaMalloc(&d_C, n_bytes);

    cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, n_bytes, cudaMemcpyHostToDevice);

    /// @brief kernel lauch with threadIdx.x as index
    /// @prototype: kernel<<<num_blocks, num_threads_per_block>>>(d_A, d_B, d_C);
    // note that both 'num_blocks' and 'num_threads_per_block' can be either int or dim3
    // we first experiment with int type
    int num_blocks = 1;
    int num_threads_per_block = N;
    
    // <<<1, N>>>
    // one block with N threads,
    // blockIdx varies in [0, 1)
    // threadIdx varies from [0, N-1)
    num_blocks = 1;
    num_threads_per_block = N;
    modulo_affine_init(h_C, N, 0, 0, 1);
    cudaMemcpy(d_C, h_C, n_bytes, cudaMemcpyHostToDevice);
    vector_add_threadidx_x_only_kernel<<<num_blocks, num_threads_per_block>>>(d_A, d_B, d_C);
    cudaMemcpy(h_C, d_C, n_bytes, cudaMemcpyDeviceToHost);
    printf("kernel launch with <<<%d, %d>>>:\n", num_blocks, num_threads_per_block);
    check_memory(h_C, N, "h_C");

    // <<<1, N>>>
    // one block with N threads,
    // blockIdx varies in [0, N-1)
    // threadIdx varies from [0, 1)
    // however, note that the called kernel uses only threadIdx.x
    // hence, for all threads we have 'idx == 0'.
    num_blocks = N;
    num_threads_per_block = 1;
    modulo_affine_init(h_C, N, 0, 0, 1);
    cudaMemcpy(d_C, h_C, n_bytes, cudaMemcpyHostToDevice);
    vector_add_threadidx_x_only_kernel<<<num_blocks, num_threads_per_block>>>(d_A, d_B, d_C);
    cudaMemcpy(h_C, d_C, n_bytes, cudaMemcpyDeviceToHost);
    printf("kernel launch with <<<%d, %d>>>:\n", num_blocks, num_threads_per_block);
    check_memory(h_C, N, "h_C");

    /// @brief kernel lauch with blockIdx.x as index,
    //      now we try the two configurations, but with 'idx = blockIdx.x' in the kernel
    // <<<1, N>>>
    num_blocks = 1;
    num_threads_per_block = N;
    modulo_affine_init(h_C, N, 0, 0, 1);
    cudaMemcpy(d_C, h_C, n_bytes, cudaMemcpyHostToDevice);
    vector_add_blockidx_x_only_kernel<<<num_blocks, num_threads_per_block>>>(d_A, d_B, d_C);
    cudaMemcpy(h_C, d_C, n_bytes, cudaMemcpyDeviceToHost);
    printf("kernel launch with <<<%d, %d>>>:\n", num_blocks, num_threads_per_block);
    check_memory(h_C, N, "h_C");

    // <<<N, 1>>>
    num_blocks = 1;
    num_threads_per_block = N;
    modulo_affine_init(h_C, N, 0, 0, 1);
    cudaMemcpy(d_C, h_C, n_bytes, cudaMemcpyHostToDevice);
    vector_add_blockidx_x_only_kernel<<<num_blocks, num_threads_per_block>>>(d_A, d_B, d_C);
    cudaMemcpy(h_C, d_C, n_bytes, cudaMemcpyDeviceToHost);
    printf("kernel launch with <<<%d, %d>>>:\n", num_blocks, num_threads_per_block);
    check_memory(h_C, N, "h_C");


    /// @brief kernel lauch with insufficient threads,
    //      only part of the results are computed
    // <<<1, N/2>>>
    num_blocks = 1;
    num_threads_per_block = N/2;
    modulo_affine_init(h_C, N, 0, 0, 1);
    cudaMemcpy(d_C, h_C, n_bytes, cudaMemcpyHostToDevice);
    vector_add_threadidx_x_only_kernel<<<num_blocks, num_threads_per_block>>>(d_A, d_B, d_C);
    cudaMemcpy(h_C, d_C, n_bytes, cudaMemcpyDeviceToHost);
    printf("kernel launch with <<<%d, %d>>>:\n", num_blocks, num_threads_per_block);
    check_memory(h_C, N, "h_C");
}