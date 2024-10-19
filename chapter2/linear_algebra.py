import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.arange(4, dtype=torch.float32)

# A_sum_dim0 = A.sum(dim=0)
# print(A_sum_dim0)
# A_sum_dim1 = A.sum(dim=1)
# print(A_sum_dim1)
# A_sum_dim01 = A.sum(dim=[0, 1])
# print(A_sum_dim01)

# 求平均数
# print(A.mean(axis=0))
# print(A.sum(axis=0) / A.shape[0])
# sum_A = A.sum(axis=1, keepdims=True)
# print(A / sum_A)

print(A.cumsum(dim=0))

print(torch.mv(A,x))