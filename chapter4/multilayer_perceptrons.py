import torch
from torch import nn
from d2l import torch as d2l
from chapter3.softmax import train_ch3,predict_ch3

batch_size = 256
# 加载训练、测试数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""初始化模型参数"""
# 输入(28*28图片)，输出(10种)，隐藏层
num_inputs, num_outputs, num_hiddens = 784, 10, 256
# 第一层参数 （隐藏层？）
# nn.Parameter() : 声明是torch的parameter
# torch.randn()：其中每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样
# （这里乘0.01，相当于标准差变成1 * 0.01 = 0.01）
# （是一个 inputs * hiddens 的矩阵）（因为一批数据是 batch_size * inputs 的矩阵）
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
# 偏差，设为全0 （为 1 * hiddens）
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# 第二层参数 （输出层）
# （是一个 hiddens * outputs 的矩阵）
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

"""激活函数"""
def relu(X):
    # torch.zeros_like(Y): 创建一个与Y形状相同的张量，其中所有元素都设置为0
    a = torch.zeros_like(X)
    return torch.max(X, a)

"""模型"""
def net(X):
    # 数据先拉成二维的矩阵
    X = X.reshape((-1, num_inputs))
    # 第一层
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    # 第二层
    return (H @ W2 + b2)

"""损失函数"""
loss = nn.CrossEntropyLoss(reduction='none')

"""训练"""
if __name__ == '__main__':
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    predict_ch3(net, test_iter)

d2l.plt.show()

""""""
""""""
