/// @brief on the purpose of this exercise:
// 1. a simple review of previous knowledge

/// @brief on what you can learn in this exercise:
// ...

/// @brief on the pipeline of this exercise:
//  1. given input array h_A, output its reversed array h_B
//  2. each thread executes one placement.


#include "utils.h"

template<typename T>
__global__ void reverse_array(T* d_A, T* d_B, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        d_B[idx] = d_A[N - idx - 1];
    }
}

int main() {
    int N = 10;
    size_t n_bytes = N * sizeof(float);

    // initialize host data
    float* h_A;
    float* h_B;
    h_A = new float[N];
    h_B = new float[N];

    modulo_affine_init(h_A, N, 5, 1, 9);
    modulo_affine_init(h_B, N, 0, 0, 1);

    printf("initialization:\n");
    check_memory(h_A, N, "h_A");
    check_memory(h_B, N, "h_B");

    // device data
    float* d_A;
    float* d_B;

    cudaMalloc(&d_A, n_bytes);
    cudaMalloc(&d_B, n_bytes);

    cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice);

    int num_threads_per_block = 3;
    int num_blocks = (N + num_threads_per_block - 1)/ num_threads_per_block;
    // num_blocks = 3;  // check that at least 4 blocks are needed
    
    printf("Running with <<<%d, %d>>>\n", num_blocks, num_threads_per_block);
    reverse_array<<<num_blocks, num_threads_per_block>>>(d_A, d_B, N);
    
    cudaMemcpy(h_A, d_A, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, n_bytes, cudaMemcpyDeviceToHost);

    printf("reversed:\n");
    check_memory(h_A, N, "h_A");
    check_memory(h_B, N, "h_B");

}