import torch
from torch import nn
from d2l import torch as d2l

"""LeNet"""
# nn.Conv2d --> （输入通道，输出通道，kernel_size= ，padding= ，stride= ）
# 没用relu和最大汇聚层是因为 那时候还没出现
net = nn.Sequential(
    # 原始输入是 2* * 28， 填充后为32 * 32
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # 均值池化层，不重叠 （变成 14 * 14）
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),  # 变成10 * 10
    nn.AvgPool2d(kernel_size=2, stride=2),  # 均值池化层，不重叠 (变成 5 * 5)
    nn.Flatten(),  # 输出为四维的，所以要拉平（第一层批量保持，后面全部拉成同一维）
    # 接下为 多层感知机
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),  # 输出为16，高宽池化后为5 * 5,所以是 16 * 5 * 5
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))  # 去掉了最后一层的高斯激活

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# 把每层拿出来，输出每层output的形状
for layer in net:
    X = layer(X)
    # 输出形状第0维都是批量大小，为1
    print(layer.__class__.__name__,'output shape: \t',X.shape)


"""在Fashion-MNIST数据集上的表现"""
# 加载数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 计算精度
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    # 如果是nn.Module 实现的话
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            # 如果没指定device，就将net的参数拿出来用它的device
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            # 如果是list得每个的device都挪一下
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 训练函数
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        # 如果为全连接层或卷积层
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # xavier 初始化，均匀分布
            nn.init.xavier_uniform_(m.weight)
    # 对每个 parameters 都应用
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    # 梯度下降
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 交叉熵损失函数 （用于解决多分类问题）
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()  # 梯度设0
            X, y = X.to(device), y.to(device)
            y_hat = net(X)  # 前向操作
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()  # 迭代
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

# 训练和评估
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

d2l.plt.show()