import numpy as np
import torch

a = np.random.rand(4,3)
print(a,type(a))

b = torch.from_numpy(a)
print(b,type(b))

print(b.numpy(),type(b.numpy()))

b.mul_(2)
print(b,type(b))
print(a,type(a))

