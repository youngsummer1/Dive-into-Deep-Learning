import torch

# torch.arange()
# x = torch.arange(12)
# print(x)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# shape
# print(x.shape)

# numel()
# print(x.numel())

# reshape()
# x = x.reshape(3,4)
# print(x)

# print(torch.zeros(2,3,4))
# print(torch.zeros((2,3,4)))

# print(torch.ones((2,3,4)))
# 和上面相似

# print(torch.randn(3,4))
# 和上面相似

# print(torch.tensor([[2,1,4,3],[1,2,3,4],[3,1,5,4]]))

x = torch.tensor([1.0, 2, 3, 4])
y = torch.tensor([2, 5, 7, 8])
# print(x+y)
# print(x-y)
# print(x*y)
# print(x/y)
# print(x**y)

# print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(torch.cat((X, Y), dim=0))
# print(torch.cat((X, Y), dim=1))
# print(X == Y)
# print(X.sum())

# a = torch.arange(3).reshape((3, 1))
# b = torch.arange(2).reshape((1, 2))
# print(a+b)

# before = id(Y)
# # Y[:] = X + Y
# Y += X
# print(id(Y) == before)

# A = X.numpy() # 框架张量 -> NumPy张量
# B = torch.tensor(A) # 框架张量 <- NumPy张量
# print(type(A))
# print(type(B))

a = torch.tensor([3.5])
# print(a.item())
# print(float(a))
# print(int(a))

# b = a
# print(id(a) == id(b))
# b = a.clone()
# print(id(a) == id(b))
