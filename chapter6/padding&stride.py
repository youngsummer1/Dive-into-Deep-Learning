import torch
from torch import nn

"""填充"""
# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1， X变成4维
    X = X.reshape((1, 1) + X.shape)  # 元组的连接，不是相加
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
# padding --> 填充
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
# 8 - 3 + 2 + 1 = 8 (padding =1填充了两行)
# print(comp_conv2d(conv2d, X).shape)

# 高度和宽度两边的填充分别为2和1
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
# 8 - 5 + 4 + 1 = 8， 8 - 3 + 2 + 1 =8
# print(comp_conv2d(conv2d, X).shape)


"""步幅"""
# 高度和宽度的步幅设置为2
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
# print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
# (8 - 3 + 0 + 3) / 3 = 2, (8 - 5 + 2 + 4) / 4 = 2
print(comp_conv2d(conv2d, X).shape)