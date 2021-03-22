# PyTorch C++ API Overview

PyTorch C++ API basic components:

- __ATen__ The foundational tensor and mathematical operation library on which all else is built.
- __Autograd__ Augments ATen with automatic differentiation.
- __C++ Frontend__ High level constructs for training and evaluation of machine learning models.
- __TorchScript__ An interface to the TorchScript JIT compiler and interpreter.
- __C++ Extensions__ A means of extending the Python API with custom C++ and CUDA routines.

### ATen

This is the library for defining the `Tensor` class and the tensor operations.

### Autograd

`Tensor` class is equipped with autograd, but only in `torch::` namespace, not `at::` namespace.

### C++ Frontend

This is a high level, pure C++ modeling interface.

### TorchScript

A programming language in its own right. It can be compiled by the Torchscript compiler.

### C++ Extensions

A powerful way to extend PyTorch from Python to C++ and CUDA. It is used to bind C++ API to Python codes.
