#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;

int SHOW_ONLY = 15;

__global__ void MatAccess(float *A, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadIdx.x == 0)
        A[i] = 0.0;
        return;

    if (i < numElements) {
        A[i] += 1.0;
    }
}

/**
 * Increase the values in the vector by 1
 * without error checks
 */
int main() {

    // Print the vector length to be used, and compute its size
    int numElements = 4;
    size_t size = numElements * sizeof(float);
    printf("[Matrix accessing test of %d elements]\n", numElements);

    // allocating memory and initialization
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *d_A = NULL; cudaMalloc((void **)&d_A, size);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 0.0;
    }

    // Copy memory: host to device
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 2; // maximum 1024, otherwise none will be executed
    int blocksPerGrid = 2;
    MatAccess<<<blocksPerGrid, threadsPerBlock>>>(d_A, numElements);
    printf("CUDA kernel launched with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    // Copy memory: device to host
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_B, d_A, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<min(SHOW_ONLY, numElements); i++) printf("%.5f ", h_A[i]); cout << "\n";
    for (int i=0; i<min(SHOW_ONLY, numElements); i++) printf("%.5f ", h_B[i]); cout << "\n";

    // Free memory
    cudaFree(d_A);
    free(h_A);
    free(h_B);

    printf("Done\n");
    return 0;
}

