import torch
from torch import nn
from d2l import torch as d2l

"""批量归一化函数"""
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    批量归一化函数
    :param X: 输入
    :param gamma: 拉伸参数，可学参数
    :param beta: 偏移参数，可学参数
    :param moving_mean: 均值的移动平均值，全局的均值
    :param moving_var: 方差的移动平均值，全局的方差
    :param eps: 噪声，防止方差为0
    :param momentum: 用来更新移动..的，通常为0.9
    :return:
    """
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():  # 即不算梯度 （评估模式）
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        # （就是直接用全局的均值和方差）（因为评估时可能没有批量，只有一张图片什么的）
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:  # 训练模式
        # 只支持全连接层和卷积层
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)  # 即把行压扁，算每列的均值
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)  # 按通道维求，(1,n,1,1)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    # 用.data 是因为会是torch.parameter?
    return Y, moving_mean.data, moving_var.data


"""创建一个BatchNorm图层"""
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        # （因为要被迭代，所以放在 nn.Parameter 里）
        self.gamma = nn.Parameter(torch.ones(shape))  # 初始为1
        self.beta = nn.Parameter(torch.zeros(shape))  # 初始为0
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y

"""在LeNet上使用"""
# net = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
#     nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
#     nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
#     nn.Linear(84, 10))  # 输出层就不加线性变化了

# 训练
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

# 查看第一个批量规范化层中学到的拉伸参数gamma和偏移参数beta
# net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))

"""简洁实现"""
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

# 训练
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())