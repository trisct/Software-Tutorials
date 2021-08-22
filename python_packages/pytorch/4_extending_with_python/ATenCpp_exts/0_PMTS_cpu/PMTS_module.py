import torch
import PMTS_cpp

class PMTSFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)

        pts, mts = PMTS_cpp.forward(a, b)
        return pts, mts
    
    @staticmethod
    def backward(ctx, grad_pts, grad_mts):
        a, b = ctx.saved_tensors
        
        grad_a, grad_b = PMTS_cpp.backward(grad_pts, grad_mts, a, b)
        return grad_a, grad_b

PMTS = PMTSFunction.apply