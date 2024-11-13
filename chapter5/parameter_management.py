import torch
from torch import nn

# 单隐藏层的多层感知机
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
# print(net(X))

"""参数访问"""
# 通过Sequential类定义模型时，可以通过索引来访问模型的任意层
# net[2] --> nn.Linear(8, 1)
# state_dict --> 该层的参数参数
# print(net[2].state_dict())
# OrderedDict([('weight', tensor([[ 0.2652, -0.0447, -0.2746,  0.1820, -0.2796,  0.1581,  0.0271, -0.0695]])), ('bias', tensor([-0.0585]))])


# 访问目标参数
# print(type(net[2].bias)) # Parameter --> 定义一个可以优化的参数
# print(net[2].bias) # 访问bias参数
# print(net[2].bias.data) # 访问bias参数的值
# print(net[2].bias.grad) # 访问bias参数的梯度 (因为没调用反向传播，所以还是None)
# <class 'torch.nn.parameter.Parameter'>
# Parameter containing:
# tensor([-0.3455], requires_grad=True)
# tensor([-0.3455])
# None


# 一次性访问所有参数
# print(net[0].named_parameters())
# print(*[(name, param.shape) for name, param in net[0].named_parameters()]) # 第一层是relu，所以没有参数
# print(*[(name, param.shape) for name, param in net.named_parameters()])

# 因此可以通过名字来访问
# print(net.state_dict()['2.bias'].data)
# print(net.state_dict()['0.weight'])


# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# print(rgnet(X)) # 输出2 * 1，是因为有两个输入样本
# print(rgnet)


"""参数初始化"""
# 内置初始化
def init_normal(m):
    """
    :param m: 一个nn.Module
    """
    if type(m) == nn.Linear:  # 全连接层
        # 加下划线 为 原地操作
        nn.init.normal_(m.weight, mean=0, std=0.01) # 对weight，均值0，标准差0.01的初始化
        nn.init.zeros_(m.bias)
# 对net里所有Module都做一遍
# net.apply(init_normal)
# print(net[0].weight.data[0], net[0].bias.data[0])

def init_constant(m):
    if type(m) == nn.Linear:
        # 对所有weight, 初始化为1
        # 但这是不行的，会导致同一层各个单元的输出都一样，相当于只有一个单元
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
# net.apply(init_constant)
# print(net[0].weight.data[0], net[0].bias.data[0])

# 对某些块应用不同的初始化方法
def init_xavier(m):
    """xavier初始化"""
    if type(m) == nn.Linear:
        # Xavier 均匀分布初始化神经网络层的权重参数，以提高网络的收敛速度和性能。
        # uniform是均值
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

# net[0].apply(init_xavier)
# net[2].apply(init_42)
# print(net[0].weight.data[0])
# print(net[2].weight.data)


# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

# net.apply(my_init)
# print(net[0].weight[:2])

"""参数绑定"""
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
# 两个share是同一块内存
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
