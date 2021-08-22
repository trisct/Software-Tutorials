import torch
import PMTS_cpp
import PMTS_cuda

class PMTSFunctionCUDA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)

        pts, mts = PMTS_cuda.forward(a.contiguous(), b.contiguous())
        return pts, mts
    
    @staticmethod
    def backward(ctx, grad_pts, grad_mts):
        a, b = ctx.saved_tensors
        
        grad_a, grad_b = PMTS_cuda.backward(grad_pts.contiguous(), grad_mts.contiguous(), a.contiguous(), b.contiguous())
        return grad_a, grad_b


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

PMTSCUDA = PMTSFunctionCUDA.apply
PMTS = PMTSFunction.apply
