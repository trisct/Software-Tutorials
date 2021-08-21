#include <iostream>
#include <vector>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

using namespace std;

int SHOW_ONLY = 15;

__global__ void AvgPool3x3_cuda(float * input, float * output, int h, int w) {
    /*
    assert H >= 3 and W >= 3 should be put in the caller of this.

    input: array of shape [H, W]
    output: array of shape [H-2, W-2] (no padding)
    shape: {H, W}
    */

    int numElements_out = (h-2) * (w-2);
    int i_out = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i_out >= numElements_out)
        return;
    
    int row_out = i_out / (w-2);
    int col_out = i_out % (w-2);

    int row_in = row_out + 1;
    int col_in = col_out + 1;

    int i_in = row_in * w + col_in;

    output[i_out] =
        input[i_in] +
        input[i_in - w] +
        input[i_in + w] +
        input[i_in + 1] +
        input[i_in - 1] +
        input[i_in - w - 1] +
        input[i_in - w + 1] +
        input[i_in + w - 1] +
        input[i_in + w + 1];
    
    output[i_out] /= 9;
}

void AvgPool3x3_caller(float * input, float * output, const vector<int> shape_in) {
    assert(shape_in.size() == 2);
    assert(shape_in[0] >= 2 && shape_in[1] >= 2);

    int threadsPerBlock = 256; // maximum 1024, otherwise none will be executed
    int blocksPerGrid = 100000;
    
    printf("CUDA kernel launched with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    AvgPool3x3_cuda<<<blocksPerGrid, threadsPerBlock>>>(input, output, shape_in[0], shape_in[1]);
}


void print_by_shape(float * A, vector<int> shape) {
    assert(shape.size() == 2);
    assert(shape[0] >= 1 && shape[1] >= 1);

    int numElements = shape[0] * shape[1];
    for (int i = 0; i < numElements; i++) {
        printf("%.8f ", A[i]);
        if ((i+1) % shape[1] == 0) {
            printf("\n");
        }
    }
}

int main() {

    // Print the vector length to be used, and compute its size
    vector<int> shape_in = {256, 100000};
    vector<int> shape_out = {shape_in[0]-2, shape_in[1]-2};

    int numElements_in = shape_in[0] * shape_in[1];
    int numElements_out = shape_out[0] * shape_out[1];
    size_t size_in = numElements_in * sizeof(float);
    size_t size_out = numElements_out * sizeof(float);

    // allocating memory and initialization
    float *h_A = (float *) malloc(size_in);
    float *h_B = (float *) malloc(size_out);
    float *d_A = NULL;
    float *d_B = NULL;
    cudaMalloc((void **)&d_A, size_in);
    cudaMalloc((void **)&d_B, size_out);

    for (int i = 0; i < numElements_in; ++i) {
        h_A[i] = rand() / (float) RAND_MAX;
    }

    // Copy memory: host to device
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, size_in, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    AvgPool3x3_caller(d_A, d_B, shape_in);
    
    // Copy memory: device to host
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_B, d_B, size_out, cudaMemcpyDeviceToHost);

    // print_by_shape(h_A, shape_in);
    // print_by_shape(h_B, shape_out);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    printf("Done\n");
    return 0;
}