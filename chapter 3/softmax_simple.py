import torch
from torch import nn
from d2l import torch as d2l
from softmax import train_ch3

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""初始化模型参数"""
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
# nn.Flatten() : 把任何维度的tensor变成2d，第0维不变，其他维展开
# nn.Linear(784, 10)： 线性层
# Sequential：一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    # 如果是一个Linear layer
    if type(m) == nn.Linear:
        # weight init 成 均值为0，标准差为0.01的随机值
        nn.init.normal_(m.weight, std=0.01)

# 即给每一层跑一遍这个函数
net.apply(init_weights);

"""重新审视Softmax的实现"""
# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

"""优化算法"""
# 学习率为0.1的小批量随机梯度下降
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

"""训练"""
num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
