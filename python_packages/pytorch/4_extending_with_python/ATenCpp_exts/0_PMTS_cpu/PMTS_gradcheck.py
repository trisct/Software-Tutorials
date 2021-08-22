import torch
import torch.autograd
from torch.autograd import gradcheck

from PMTS_module import PMTS as pmts

module_name = "PMTS"
print(f"\n############ starting grad check of module {module_name} ############")

# only double is precise enough for gradcheck
a = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
b = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

test = gradcheck(pmts, (a, b), eps=1e-6, atol=1e-4)
print(f"grad check passed: {test}")
print(f"############## grad check of module {module_name} done ##############\n")
c = pmts(a, b)
