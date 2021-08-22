import torch
import torch.autograd
from torch.autograd import gradcheck

from PMTS_module import PMTSCUDA as pmtscuda
from PMTS_module import PMTS as pmts
import time

module_name = "PMTS"
print(f"\n############ starting time test of module {module_name} ############")

# only double is precise enough for gradcheck
a = torch.randn(500000, requires_grad=True)
b = torch.randn(500000, requires_grad=True)

iters = 100

# cpu
time_s = time.time()
for i in range(iters):
    c, d = pmts(a, b)
    z = (c + d).sum()
    z.backward()
time_e = time.time()
print(f"CPU          time test: {(time_e - time_s) / iters} s per iter.")

# gpu, aten backend
a = a.cuda()
b = b.cuda()
time_s = time.time()
for i in range(iters):
    c, d = pmts(a, b)
    z = (c + d).sum()
    z.backward()
time_e = time.time()
print(f"GPU (ATen)   time test: {(time_e - time_s) / iters} s per iter.")

# gpu, custom cuda kernels
time_s = time.time()
for i in range(iters):
    c, d = pmtscuda(a, b)
    z = (c + d).sum()
    z.backward()
time_e = time.time()
print(f"GPU (custom) time test: {(time_e - time_s) / iters} s per iter.")

print(f"############## time test of module {module_name} done ##############\n")
