# Tensor toturial

# This teaches you how to define and operate on tensors.

# Tensor (in computer science) is merely a another name for high-dimensional arrays.

import torch

# This creates a tensors as specified by yourself
a = torch.tensor([[1,3,5,4],
                  [4,7,9,-2],
                  [10,-2,1,0]])

# This creates a tensor of specified shape whose terms are generated by the uniform distribution on [0,1]
b = torch.rand(3,4)

# This creates a tensor of specified shape whose terms are generated by the standard normal distribution
c = torch.randn(3,4)

# This creates a tensor of specified shape with all zeros
d = torch.rand(3,4)

# This creates a tensor of specified shape with all ones
e = torch.ones(3,4)

# This creates a tensor like a unit matrix
f = torch.eye(3)

# Tensors of the same shape can be added, subtracted, multiplied and divided (elementwise)
g = (a+b)*c-d

print('a =',a)
print('b =',b)
print('c =',c)
print('d =',d)
print('e =',e)
print('f =',f)
print('g =',g)
