import torch
import torch.autograd
from torch.autograd import gradcheck

# element-wise multiplication

# Inherit from Function
class ElementWiseMultiplication(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        assert a.shape == b.shape


        ctx.save_for_backward(a, b)
        print(f"[In forward] a.requires_grad = {a.requires_grad}, b.requires_grad = {b.requires_grad}")

        return a * b

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            grad_a = grad_output * b
        if ctx.needs_input_grad[1]:
            grad_b = grad_output * a

        print(f"[In backward] a.requires_grad = {a.requires_grad}, b.requires_grad = {b.requires_grad}")
        return grad_a, grad_b
        

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
em = ElementWiseMultiplication.apply
a, b = (torch.randn(30, 20, dtype=torch.double, requires_grad=True),
        torch.randn(30, 20, dtype=torch.double, requires_grad=True))
test = gradcheck(em, (a, b), eps=1e-6, atol=1e-4)
print(test)

a = a.cuda().float()
b = b.cuda().float()

z = em(a, b)