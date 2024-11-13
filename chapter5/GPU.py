import torch
from torch import nn

# ？我怎么看不到
# torch.device('cpu')  # CPU
# print(torch.device('cuda'))  # 第0块GPU
# print(torch.device('cuda:1'))  # 第1块GPU

# 查询可用gpu的数量
# print(torch.cuda.device_count())

"""允许我们在不存在所需所有GPU的情况下运行代码"""
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

# print(try_gpu())
# print(try_gpu(10))
# print(try_all_gpus())

# 查询张量所在的设备
# x = torch.tensor([1, 2, 3])
# print(x.device)

X = torch.ones(2, 3, device=torch.device('cuda'))
# print(X)

"""神经网络与GPU"""
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
# print(net(X))

# 确认模型参数存储在同一个GPU上
print(net[0].weight.data.device)