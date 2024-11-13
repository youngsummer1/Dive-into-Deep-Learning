import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
from chapter3.softmax import train_epoch_ch3

"""生成数据集"""
max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

# 特征
features = np.random.normal(size=(n_train + n_test, 1))
# 特征打乱
np.random.shuffle(features)
# 多项式特征 , 为 n_train + n_test * max_degree 的矩阵
# （如features是x ， 则为 [1,x,x^2,x^3 ,...,x^19]）
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    # math.gamma 阶乘
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w) # 多项式特征 * 多项式系数 （点乘，广播）
labels += np.random.normal(scale=0.1, size=labels.shape) # 加入随机噪声

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

# print(features[:2])
# print(poly_features[:2, :])
# print(labels[:2])

"""对模型进行训练和测试"""
# 评估
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape) # 形状
        l = loss(out, y) # 求损失
        metric.add(l.sum(), l.numel())
    # 在整个数据集算整个损失
    return metric[0] / metric[1]

#训练函数
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none') #
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它 （即 w0 * x^0 ?）
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False)) # 线性网络
    batch_size = min(10, train_labels.shape[0]) # 批大小 （如果样本数 > 10）
    # load 数据集
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    # 显示
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            # 打印曲线变化
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

"""正常"""
# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3! （只设置了前四列的权重）
# 每一项都作为一个特征，多项式就相当于线性回归 （所以说模型和数据是匹配的？）
# train(poly_features[:n_train, :4], poly_features[n_train:, :4],
#       labels[:n_train], labels[n_train:])

"""欠拟合"""
# 从多项式特征中选择前2个维度，即1和x
# （模型容量不够，只有两个参数，导致欠拟合）（该学的没学）
# train(poly_features[:n_train, :2], poly_features[n_train:, :2],
#       labels[:n_train], labels[n_train:])

"""过拟合"""
# 从多项式特征中选取所有维度
# （模型容量过大，导致过拟合）（[4:]都为0，都是噪声）（学了堆不该学的）
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=2500)

d2l.plt.show()