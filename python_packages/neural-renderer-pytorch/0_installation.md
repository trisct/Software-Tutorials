# Neural Renderer PyTorch

The installation may fail for new PyTorch versions for the following reasons:

```
/home/user/.dev_apps/neural_renderer_pytorch/neural_renderer/cuda/load_textures_cuda.cpp:13:23: error: ‘AT_CHECK’ was not declared in this scope; did you mean ‘DCHECK’?
       13 | #define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
          |                       ^~~~~~~~
```

Try to use the `at_assert_fix` branch instead of the master branch:
```
git clone https://github.com/prashantraina/neural_renderer_pytorch/tree/at_assert_fix
```

Then go to each cpp file in `neural_renderer_pytorch/neural_renderer/cuda/` and change all `AT_CHECK` to `AT_ASSERT`. Finally,
```
pip install -v -e .
```
