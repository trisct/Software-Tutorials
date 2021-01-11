# Concepts

### Official Guide

CUDA C++ programming official guide can be found at https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf.

### Basic Concepts and Programming Model

#### Concepts Mentioned in This Chapter

- streaming multiprocessor (SM).
- blocks
- threads
- kernel
- execution configuration syntax
- thread index `threadIdx`
- block index `blockIdx`
- block dimension `blockDim`
- shared memory
- compute capability (SM version)
- heterogenous programming
- PTX code
- cubin object
- host code
- device code

#### Scalable Programming Model

__Automatic scalability__ CUDA programming model is an easy-to-learn programming model that automatically scales the utilization of manycore GPUs when the number of cores vary.

Three key abstractions:
 - a hierarchy of thread groups
 - shared momeries
 - barrier synchronization

#### Kernel execution

A _kernel_ is a function that is executed in parallel by different CUDA threads. Each thread executes the kernel once. Once can identify a kernel by built-in variables `threadIdx`, `blockIdx` and `blockDim`.

#### Compiling

The compilation is done by `nvcc`.

__Binary compatibility__ `-code` specifies the targeted architecture. For example, `-code=sm_35` produces binary code for devices of compute capability 3.5. Compute capabilities are backward compatible within minor revisions.

__PYX compatibility__ `-arch` specifies the compute capability when compiling C++ to PTX code.

__Application compatibility__ One can specify at compile time multiple SM versions using `-gencode`

__Machine length compatibility__ A 32-bit `nvcc` can also compile device code in 64-bit mode by using `-m64`. Conversely, one can use `-m32`.