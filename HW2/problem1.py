import torch

temp = torch.FloatTensor(list(range(9)))
print("1.a")
print(temp.size())
print(temp.storage_offset())
print(temp.stride())

#1.b
# torch.cos(input, *, out=None) -> Tensor
# example torch.cos(temp)

#1.c
# torch.sqrt(input, *, out=None) -> Tensor
# torch.sqrt(temp)

temp = torch.cos(temp)
temp = torch.sqrt(temp)
print(temp)
