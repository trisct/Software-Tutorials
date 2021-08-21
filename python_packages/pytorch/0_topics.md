# Topics to Study

[TOC]

## Elements in PyTorch Autograd

- [ ] [Function](https://pytorch.org/docs/stable/autograd.html#function)

## Extending PyTorch

### Resources

- [ ] [Extending PyTorch](https://pytorch.org/docs/stable/notes/extending.html)

- [ ] [Double backward with custom functions](https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)

- [ ] Using the PyTorch C++ frontend: https://pytorch.org/tutorials/advanced/cpp_frontend.html
- [ ] Custom C++ and CUDA extensions: https://pytorch.org/tutorials/advanced/cpp_extension.html
- [ ] Autograd in C++ frontend: https://pytorch.org/tutorials/advanced/cpp_autograd.html
- [ ] Fusing convolution and batch norm using custom function: https://pytorch.org/tutorials/intermediate/custom_function_conv_bn_tutorial.html

### Levels of Extension

The following are different levels of extensions:

- custom python `torch.autograd.Function` class with tensor operations in python
- custom python `torch.autograd.Function` class with tensor operations in C++ with ATen library
- custom python `torch.autograd.Function` class with tensor operations in custom CUDA kernels

Note that using the ATen library allows using GPU as well, but can only be used with the tensor APIs in the ATen library.
