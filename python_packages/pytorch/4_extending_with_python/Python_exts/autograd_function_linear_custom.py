import torch
import torch.autograd
from torch.autograd import gradcheck

# Inherit from Function
class LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        """
        simple input -> output format
        """
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        print(f"[In forward] input.requires_grad = {input.requires_grad}, weight.requires_grad = {weight.requires_grad}")
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        """
        input: each is the grad w.r.t. the output(s) of forward
        output: each is the grad w.r.t. the input(s) of forward
        """
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        print(f"[In backward] input.requires_grad = {input.requires_grad}, weight.requires_grad = {weight.requires_grad}")
        return grad_input, grad_weight, grad_bias
        

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
linear = LinearFunction.apply
input = (torch.randn(20, 20, dtype=torch.double, requires_grad=True),
         torch.randn(30, 20, dtype=torch.double, requires_grad=True))
test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
print(test)
