#include <iostream>
#include <vector>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

using namespace std;

int SHOW_ONLY = 15;

__global__ void innerprod_cuda(float * input_a, float * input_b, float * output, int n) {
    // output: inner product of input_a and input_b
    // n: length of input_a and input_b

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i >= n)
        return;
    
    (*output) += input_a[i] * input_b[i];
    __syncthreads();
}

void innerprod_cpu(float * input_a, float * input_b, float * output, int n) {
    // output: inner product of input_a and input_b
    // n: length of input_a and input_b

    for (int i = 0; i < n; i++) {
        (*output) += input_a[i] * input_b[i];
    }
}

__global__ void innerprod_cuda_sinthread(float * input_a, float * input_b, float * output, int n) {
    // output: inner product of input_a and input_b
    // n: length of input_a and input_b

    for (int i = 0; i < n; i++) {
        (*output) += input_a[i] * input_b[i];
    }
}

// __global__ float innerprod_cuda_sinthread_ret(float * input_a, float * input_b, float * output, int n) {
//     // output: inner product of input_a and input_b
//     // n: length of input_a and input_b

//     for (int i = 0; i < n; i++) {
//         (*output) += input_a[i] * input_b[i];
//     }
//     return *output;
// }

int main() {

    // Print the vector length to be used, and compute its size
    
    int n = 512000;
    size_t n_bytes = n * sizeof(float);

    // allocating memory and initialization
    float *h_A = (float *) malloc(n_bytes);
    float *h_B = (float *) malloc(n_bytes);
    float *d_A = NULL;
    float *d_B = NULL;
    cudaMalloc((void **)&d_A, n_bytes);
    cudaMalloc((void **)&d_B, n_bytes);

    for (int i = 0; i < n; ++i) {
        h_A[i] = rand() / (float) RAND_MAX;
        h_B[i] = rand() / (float) RAND_MAX;
    }

    float * res_cuda;
    cudaMalloc(((void **)&res_cuda), sizeof(float));
    float res_cpu = 0.0;
    float res_cuda_copy = 0.0;
    cudaMemcpy(res_cuda, &res_cpu, sizeof(float), cudaMemcpyHostToDevice);

    // Copy memory: host to device
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice);

    // Launch cuda kernel
    //innerprod_cuda_sinthread<<<1, 1>>>(d_A, d_B, res_cuda, n);
    innerprod_cpu(h_A, h_B, &res_cpu, n);
    cudaMemcpy(&res_cuda_copy, res_cuda, sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    printf("cpu:  %.12f\ncuda: %.12f\n", res_cpu, res_cuda_copy);
    // printf("cpu:      %.12f\ncuda:     %.12f\ncuda_ret: ", res_cpu, res_cuda_copy, res_cuda_ret);

    printf("Done\n");
    return 0;
}