import torch
from torch import nn
from d2l import torch as d2l
from chapter3.softmax import train_ch3
"""模型"""
# 参数一：flatten成二维的
# 参数二：线性层
# 参数三：激活函数
# 参数四：输出层
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

batch_size, lr, num_epochs = 256, 0.1, 10
# 当 reduction='none' 时，表示损失函数会返回每个样本的损失值，而不是整个批次的损失值的平均或总和。
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

d2l.plt.show()