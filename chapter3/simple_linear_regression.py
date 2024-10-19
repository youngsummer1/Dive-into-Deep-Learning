import numpy as np
import torch
# 有一些处理数据的模块
from torch.utils import data
from d2l import torch as d2l
# nn是神经网络的缩写
from torch import nn

"""生成数据集"""
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 通过框架的函数
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

"""读取数据集"""
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """
    构造一个PyTorch数据迭代器
        is_train: 是不是数据集，是的话要打乱
    """
    # TensorDataset:把输入的两类数据进行一 一对应；
    dataset = data.TensorDataset(*data_arrays) # * 表示对元组解开入参，(features, labels)
    # DataLoader：重新排序， 每次从dataset随机挑选batch_size个数据
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # shuffle 是否打乱

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 演示一下，得到一个 X 和 y
print(next(iter(data_iter)))

"""定义模型"""
# Sequential一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
# 同时以神经网络模块为元素的有序字典也可以作为传入参数。
# 就，是一个list of layers?
net = nn.Sequential(nn.Linear(2, 1)) # 2 --输入的维度，1 --输出的维度

"""初始化模型参数"""
# NET[0]是容器中第1个模型：对应2个输入，1个输出的线形网络层
# weight --访问到权重w ，data --真实data ，normal_ --使用正态分布替换掉data的值
net[0].weight.data.normal_(0, 0.01)
# bias --偏差b， fill_(0) --使用0替换掉data的值
net[0].bias.data.fill_(0)

"""定义损失函数"""
# MSELoss类: 计算均方误差，
#       默认情况下，它返回所有样本损失的平均值
loss = nn.MSELoss()

"""定义优化算法"""
# 实例化一个SGD实例
# net.parameters(): net内的所有参数，包括w、b
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

"""训练"""
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        # net 自带了些参数，所以不用传w和b
        l = loss(net(X) ,y) # y --真实值
        # 优化器 梯度清零
        trainer.zero_grad()
        # pytorch已经做了sum()，所以直接backward()即可
        l.backward()
        # 进行模型更新
        # 通过优化器，改变参数，net模型也发生改变，最后计算全部输入值的预测值与真实值的损失大小
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')