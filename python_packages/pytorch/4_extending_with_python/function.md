# Function

## Note 1: Basics on `autograd.Function`

When `forward` of some `autograd.Function` object is called, tensors will be set to not requiring gradient to avoid gradient engine in the operations defined within the `forward` and `backward` call?

> [`forward()`](https://pytorch.org/docs/stable/generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward) - the code that performs the operation. It can take as many arguments as you want, with some of them being optional, if you specify the default values. All kinds of Python objects are accepted here. `Tensor` arguments that track history (i.e., with `requires_grad=True`) will be converted to ones that don’t track history before the call, and their use will be registered in the graph.

## Note 2: Levels of Extension

The following are different levels of extensions:

- custom python `torch.autograd.Function` class with tensor operations in python
- pure `C++` interface implementation with ATen
- custom python `torch.autograd.Function` class with tensor operations in C++ with ATen library
- custom python `torch.autograd.Function` class with tensor operations in custom CUDA kernels

Note that using the ATen library allows using GPU as well, but can only be used with the tensor APIs in the ATen library.