/// @brief on the purpose of this exercise:
// 1. understand how to allocate memory on cuda devices
// 2. understand how to copy data between host and device
// 3. understand how to deallocate memory on cuda devices

/// @brief on the functions you can learn in this exercise:
// 1. cudaMalloc, cudaFree
// 2. cudaMemcpy

/// @brief on the pipeline of this exercise:
// 1. allocate arrays h_A, h_B on host and d_A, d_B on device
// 2. initialize h_A and h_B
// 3. copy the data from h_A -> d_A -> d_B -> h_B
// 4. deallocate all allocated memory

#include <iostream>
#include <cstdio>
#include "utils.h"

int main() {
    
    int N = 8;
    size_t size = N * sizeof(float);
    
    printf("number elems: %d, byte per elem: %d, total size (in bytes): %d\n",
            N, int(size / N), int(size));

    
    // Allocate input vectors h_A and h_B in host memory
#ifdef C_ALLOC
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
#else
    float* h_A = new float[N];
    float* h_B = new float[N];
#endif

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = N - i;
    }

    // check copied memory:
    printf("after initialization:\n");
    check_memory(h_A, N, "h_A");
    check_memory(h_B, N, "h_B");
    

    // Allocate vectors in device memory
    /// @prototype:
    //      ​cudaError_t cudaMalloc(void** buffer, size_t size);
    /// @param void** buffer:   the address of the pointer for allocated data
    /// @param size_t size:     size of allocated memory in bytes
    float* d_A; cudaMalloc(&d_A, size);
    float* d_B; cudaMalloc(&d_B, size);
    
    // Copy vectors from host memory to device memory
    /// @prototype:
    //      cudaError_t cudaMemcpy(void* dst, const void* src, size_t size, enum cudaMemcpyKind kind)
    /// @param void* dst:       pointer to destination memory
    /// @param void* src:       pointer to source memory
    /// @param size_t* size:    size of memory to copy
    /// @param enum cudaMemcpyKind* kind:    type of transfer
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, d_A, size, cudaMemcpyDeviceToDevice);   
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);     // Copy result from device memory to host memory
    
    // check copied memory:
    printf("after copying:\n");
    check_memory(h_A, N, "h_A");
    check_memory(h_B, N, "h_B");

    // check copied memory:
    cudaMemcpy(h_B + 1, d_B, size, cudaMemcpyDeviceToHost);
    printf("after copying with an offset:\n");
    check_memory(h_A, N, "h_A");
    check_memory(h_B, N, "h_B");

    // Free device memory
    /// @prototype:
    //      cudaError_t cudaFree(void* device_ptr) 	
    cudaFree(d_A);
    cudaFree(d_B);

    // Free host memory
#ifdef C_ALLOC
    free(h_A);
    free(h_B);
#else
    delete [] h_A;
    delete [] h_B;
#endif

    pause_terminal();
}
