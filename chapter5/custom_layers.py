import torch
import torch.nn.functional as F
from torch import nn

"""不带参数的层"""
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

# layer = CenteredLayer()
# print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
# 均值为0
# （因为处理浮点数，所以得到的是很小的非零数）
# print(Y.mean())

"""带参数的层"""
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        """
        :param in_units: 输入大小
        :param units: 输出大小
        """
        super().__init__()
        # 参数要是 nn.Parameter类的实例
        self.weight = nn.Parameter(torch.randn(in_units, units)) # 输入大小 * 输出大小 的正态分布初始化
        self.bias = nn.Parameter(torch.randn(units,)) # 一般用zero
    def forward(self, X):
        # matmul --> 矩阵乘法
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
# print(linear.weight)

# 使用自定义层直接执行前向传播计算
print(linear(torch.rand(2, 5)))
# 使用自定义层构建模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))