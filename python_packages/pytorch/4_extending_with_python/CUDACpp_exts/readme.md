# Extending PyTorch with Custom CUDA Kernels

Generally, a project for extending PyTorch with custom CUDA kernels can be structured as follows.

```
project_folder/
	functions.h
	functions_cpp_interfaces.cpp
	functions_cuda_kernels.cu
	function_py_interfaces.py
	setup.py
```

Note that the `cu` and `cpp` file(s) must NOT be named the same. Otherwise a `multiple definition of xxx` error appears at compilation, since `setuptools` cannot distinguish files with different extension names.

### Header File

In the header file you can declare functions to be used.

### CPP Interface File

Same as before, you define the CPP interfaces with pybind11 specifications at the end.

### CUDA Kernel File

You implement custom CUDA kernels here.

### Python Module Setup File

In `setup.py` you specify the relative information about the custom cpp functions as a python module. Run `python setup.py install --user` to install this module to enable importing it in Python. You need to add the `cpp` files and the `cu` files as source files. The extension type is `CUDAExtension`.

### Python Interface File

Same as before, you inherit a `torch.autograd.Function` class where you call the custom forward, backward functions.
