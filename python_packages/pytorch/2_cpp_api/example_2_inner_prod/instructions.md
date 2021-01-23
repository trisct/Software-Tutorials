# C++ Extension Example 1: LLTM

This is an exact replica of https://pytorch.org/tutorials/advanced/cpp_extension.html, with some insightfule notes. This example may be too difficult. Hence I created another simpler example [alternate_resign](../example_2_alternate_resign/instructions.md).

### File Structure, Compilation and Application.

The file structure should look like
```
lltm-extension/
    lltm.cpp
    setup.py
```

To compile, do
```
cd lltm-extension
python setup.py install
```
or

```
cd lltm-extension
python setup.py install --user
```
if you don't have the root access.

After this is done, the new package can be loaded in python systemwide. In python:

```python
import torch
import lltm_cpp

help(lltm_cpp.forward)
help(lltm_cpp.backward)
```

### Key Points Explained

#### In `setup.py`
This can be seen as a configuration file for python and cpp binding.

#### In `lltm.cpp`

__Headers__ The header `<torch/extension>` must be included to access C++ APIs and the extension binding module.

__Functions__ Five functions in total are defined.

 - `torch::Tensor d_sigmoid(torch::Tensor z)`: Derivative of sigmoid at `z`.
 - `torch::Tensor d_tanh(torch::Tensor z)`: Derivative of tanh at `z`.
 - `torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0)`: Derivative of elu at `z`.
 - `std::vector<at::Tensor> lltm_forward(torch::Tensor input, ...)`: Inputs the parameters required for a forward call and returns the output, a list of tensors.
 - `std::vector<torch::Tensor> lltm_backward(torch::Tensor grad_h, ...)`: Inputs the parameters required for a backward call and returns the gradients.

__Pybind11 configs__ At the end of the file, the part
```c++
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}
```
tells pybind that the funtion `lltm_forward` will be binded to the name `forward` in python (under the namespace (module name) defined in `setup.py`, namely `lltm_cpp`). The same is true for `lltm_backward`.

#### In `use_lltm.py`

__Imports__ Must import `torch` before the custom module.

__Function wrapping__ The custom forward and backward functions should be methods of a class inherited from `torch.autograd.Function`.

<span style="color:red">Q: What is `ctx`?</span>
<span style="color:red">Q: Why the staticmethod decorator?</span>

- The input parameters of `backward` are the gradients of the output of `forward`, propogated back from the next layer.
- The output parameters of `backward` are the gradients of the input of `forward`, to propogate back to the previous layer.

More information of extending the `Function` and the `Module` classes can be found in https://pytorch.org/docs/master/notes/extending.html.

__Module definition__ To apply the extended funtion class, use `LLTMFunction.apply`. This will automatically compute the gradients.