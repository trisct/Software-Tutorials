# Extending PyTorch with C++/ATen

Generally, a project for extending PyTorch with C++/ATen is structured as follows.

```
project_folder/
	function_cpp_interface.cpp
	setup.py
	function_py_interface.py
```

### CPP Interface File

In the CPP interface file you implement the core forward and backward functions to be called, e.g.,

```c++
#include <torch/extension.h>
#include <iostream>

torch::Tensor my_forward(torch::Tensor input) {
    ...
    auto result = ...;
    return result;
}

torch::Tensor my_backward(torch::Tensor grad) {
    ...
    auto result = ...;
    return result;
}
```

with a pybind11 specification at the end:

```c++
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &my_forward, "custom forward");
  m.def("backward", &my_backward, "custom backward");
}
```

### Python Module Setup File

In `setup.py` you specify the relative information about the custom cpp functions as a python module. Run `python setup.py install --user` to install this module to enable importing it in Python.

### Python Interface File

In the Python interface file, you inherit a `torch.autograd.Function` class where you call the custom forward, backward functions:

```python
import torch
import CustomModule	# your module name defined in setup.py

class CustomFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ...
        output = CustomModule.forward(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        ...
        grad_input = CustomModule.backward(grad_output)
        return grad_input
```

Note that in C++/ATen extensions, you can use cuda tensors as well. Just put them to GPU when created in Python.
