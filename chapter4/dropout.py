import torch
from torch import nn
from d2l import torch as d2l
from chapter3.softmax import train_ch3

def dropout_layer(X, dropout):
    """以 dropout 的概率丢弃张亮输入 X 中的元素"""
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    # torch.rand() 会生成[0,1)均匀分布的...
    # 这里是对矩阵每个元素都比一次，返回一组boolean
    # 大于 dropout 的就是被选中的
    mask = (torch.rand(X.shape) > dropout).float()
    # 这样做远远比 X[mask] 快，性能消耗小
    return mask * X / (1.0 - dropout)

X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
# print(X)
# print(dropout_layer(X, 0.))
# print(dropout_layer(X, 0.5))
# print(dropout_layer(X, 1.))

"""定义具有两个隐藏层的多层感知机，每个隐藏层包含256个单元"""
#定义模型参数
# (输入， 输出， 隐藏层单元数)
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
#定义模型
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        """
        :param is_training: 判断是在测试还是训练
        """
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        # 两个隐藏层
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        # 输出层
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        # 激活函数 ， 使用relu
        self.relu = nn.ReLU()

    def forward(self, X):
        # 第一个隐藏层输出
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        # 输出层不做 dropout
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

"""训练和测试"""
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# trainer = torch.optim.SGD(net.parameters(), lr=lr)
# train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

"""简洁实现"""
net = nn.Sequential(nn.Flatten(),  # 拉平
        nn.Linear(784, 256),  # 第一个全连接层
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        # 在这放ReLU前面后面无关是因为DRopout随机删除的跟ReLU输出值没有关系
        nn.Dropout(dropout1),
        nn.Linear(256, 256),  # 第二个全连接层
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


d2l.plt.show()