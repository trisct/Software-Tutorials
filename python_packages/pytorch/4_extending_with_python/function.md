# Function

## Note 1

When `forward` of some `autograd.Function` object is called, tensors will be set to not requiring gradient to avoid gradient engine in the operations defined within the `forward` and `backward` call?

> [`forward()`](https://pytorch.org/docs/stable/generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward) - the code that performs the operation. It can take as many arguments as you want, with some of them being optional, if you specify the default values. All kinds of Python objects are accepted here. `Tensor` arguments that track history (i.e., with `requires_grad=True`) will be converted to ones that don’t track history before the call, and their use will be registered in the graph.

