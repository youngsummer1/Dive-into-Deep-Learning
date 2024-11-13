import torch
from torch import nn
from d2l import torch as d2l

"""二维互相关运算"""
def corr2d(X, K):  #@save
    """
    计算二维互相关运算 (cross-correlation)
        X: 输入
        K: 核矩阵
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 算点积（相同位置相乘，再相加）
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# print(corr2d(X, K))

"""二维卷积层"""
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        """
        :param kernel_size: 核大小
        """
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

"""边缘检测"""
# 构造一个 6 * 8 像素的黑白图像
X = torch.ones((6, 8))
X[:, 2:6] = 0
# print(X)

# 构造一个卷积核K
# 当进行互相关运算时，如果水平相邻(垂直不行)的两元素相同，则输出为零，否则输出为非零
K = torch.tensor([[1.0, -1.0]])

# 把 1 -> 0 和 0 -> 1 的边缘检测出来
Y = corr2d(X, K)
# print(Y)

# 垂直的 检查不出来
# print(corr2d(X.t(), K))

"""学习卷积核"""
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()  # 梯度清零
    l.sum().backward()
    # 迭代卷积核 （梯度下降）
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1, 2)))