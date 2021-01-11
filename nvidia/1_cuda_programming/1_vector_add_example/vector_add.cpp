// kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    float A[20], B[20], C[20];
    VecAdd<<<1, N>>>(A, B, C);
}