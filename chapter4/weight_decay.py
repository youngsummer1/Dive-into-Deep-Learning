# %matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l

"""生成一些数据"""
# 训练数据小， 特征维度大， 很容易发生过拟合
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
# 权重 & 偏移
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
# 生成人工数据集
train_data = d2l.synthetic_data(true_w, true_b, n_train)
# 读取数组变成迭代器
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

"""初始化模型参数"""
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True) # 正态
    b = torch.zeros(1, requires_grad=True) # 标量
    return [w, b]

"""定义L2范数惩罚"""
def l2_penalty(w):
    """L2范数的平方"""
    return torch.sum(w.pow(2)) / 2
    """换成L1试试"""
    # return torch.sum(torch.abs(w))
"""训练"""
def train(lambd):
    """
    :param lambd: 超参数，惩罚项

    """
    # 初始化
    w, b = init_params()
    # net 为简单的线性回归， loss 平方损失函数
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # 训练轮数 学习率
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size) # sgd里应该写了梯度清零
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    # 这里没有平方，就只是L2范数
    print('w的L2范数是：', torch.norm(w).item())

"""忽略正则化"""
# train(lambd=0)
"""使用权重衰减"""
# train(lambd=10)


"""简单实现"""
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    # 均方损失函数
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    # weight decay --> 权重衰退, wd --> lambda
    # （将惩罚项直接写在训练算法里）
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())

"""忽略正则化"""
train_concise(wd=0)
"""使用权重衰减"""
# train_concise(wd=10)

d2l.plt.show()