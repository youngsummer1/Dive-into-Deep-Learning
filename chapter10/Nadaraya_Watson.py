import torch
from torch import nn
from d2l import torch as d2l

"""生成数据集"""
n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本
# print(x_train)
def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, 5, 0.1)  # 测试样本 (0-5之间，步长0.1)
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
# print(n_test)


def plot_kernel_reg(y_hat):
    """绘制所有的训练样本（样本由圆圈表示）"""
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);

"""平均汇聚"""
# 基于平均汇聚来计算所有训练样本输出值的平均值
# y_hat = torch.repeat_interleave(y_train.mean(), n_test)
# plot_kernel_reg(y_hat)

"""非参数注意力汇聚"""
# X_repeat的形状:(n_test,n_train),
# 每一行都包含着相同的测试输入（例如：同样的查询）
# X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
# attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
# (50,50) * (50,1) = (50,1)?
# y_hat = torch.matmul(attention_weights, y_train)
# plot_kernel_reg(y_hat)

# 观察注意力的权重
# 因为两个输入都是经过排序的，因此由观察可知“查询-键”对越接近， 注意力汇聚的注意力权重就越高
# d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
#                   xlabel='Sorted training inputs',
#                   ylabel='Sorted testing inputs')


"""带参数注意力汇聚"""
# 批量矩阵乘法
# X = torch.ones((2, 1, 4))
# Y = torch.ones((2, 4, 6))
# print(torch.bmm(X, Y).shape)

# 可以使用小批量矩阵乘法来计算小批量数据中的加权平均值
# weights = torch.ones((2, 10)) * 0.1
# values = torch.arange(20.0).reshape((2, 10))
# (2,1,10) * (2,10,1) = (2,1,1)
# print(torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)))


# 定义模型
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 创建权重参数
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        # 我的理解： 查询个数 -- 即要查哪几个数，将它变化之后，每行都是相同的一个要查询的数，用来和keys的每个值去计算远近关系
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        # 我的理解：查询个数 —— 即要查哪几个数，所以不需要变化，所以给attention和values加个维度让它们去乘
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

"""训练"""
# 在带参数的注意力汇聚模型中
# 任何一个训练样本的输入都会和除自己以外的所有训练样本的“键－值”对进行计算， 从而得到其对应的预测输出
# （所以去除了对角线上的数）

# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)  (把对角线上的去掉了)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
print(keys.shape)
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))


# 训练,使用平方损失函数和随机梯度下降
# net = NWKernelRegression()
# loss = nn.MSELoss(reduction='none')
# trainer = torch.optim.SGD(net.parameters(), lr=0.5)
# animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
#
# for epoch in range(5):
#     trainer.zero_grad()
#     l = loss(net(x_train, keys, values), y_train)
#     l.sum().backward()
#     trainer.step()
#     print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
#     animator.add(epoch + 1, float(l.sum()))


# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
# keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
# values = y_train.repeat((n_test, 1))
# y_hat = net(x_test, keys, values).unsqueeze(1).detach()
# plot_kernel_reg(y_hat)


# 曲线在注意力权重较大的区域变得更不平滑
# d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
#                   xlabel='Sorted training inputs',
#                   ylabel='Sorted testing inputs')

d2l.plt.show()