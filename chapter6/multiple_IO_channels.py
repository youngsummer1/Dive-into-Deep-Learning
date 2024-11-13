import torch
from d2l import torch as d2l

"""多输入通道"""
def corr2d_multi_in(X, K):
    """
    2D互相关运算，多个输入通道运算
        X：输入，3维
        K：卷积核，3维
    """
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    # X和K都是3维的，zip可以每次取出最外层的一个通道
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
# print(corr2d_multi_in(X, K))

"""多输出通道"""
def corr2d_multi_in_out(X, K):
    """
    2D互相关运算，多个输出通道运算
        X：输入，3维
        K：卷积核，4维，输出通道 * 输入通道 * 高 * 宽
    """
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    # torch.stack():沿着一个新维度对输入张量序列进行连接，序列中所有的张量都应该为相同形状
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

# 具有3个输出通道的卷积核
K = torch.stack((K, K + 1, K + 2), 0)
# print(K.shape)
# print(corr2d_multi_in_out(X, K))  # 3 * 2 * 2 的输出

"""1 * 1 卷积"""
def corr2d_multi_in_out_1x1(X, K):
    """用全连接实现的 1*1 多输入输出通道 互相关操作"""
    c_i, h, w = X.shape
    c_o = K.shape[0]
    # 高宽 拉平
    X = X.reshape((c_i, h * w))
    # 拉平 （因为是 1 * 1，所以可以这样拉）
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)  # 全连接实现
Y2 = corr2d_multi_in_out(X, K)  # 正常实现
# 说明差别不大
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6