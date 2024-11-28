import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from language_model import load_data_time_machine
from RNN import predict_ch8,train_ch8

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

"""定义模型"""
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

# 使用张量来初始化隐状态，它的形状是（隐藏层数，批量大小，隐藏单元数）
state = torch.zeros((1, batch_size, num_hiddens))
# print(state.shape)

# 通过一个输入和一个隐状态，我们就可以用更新后的隐状态计算输出
# X 形状 （时间步，批量大小，词表大小）
# X = torch.rand(size=(num_steps, batch_size, len(vocab)))
# rnn_layer的“输出”（Y）不涉及输出层的计算： 它是指每个时间步的隐状态，这些隐状态可以用作后续输出层的输入。
# Y, state_new = rnn_layer(X, state)
# Y 形状 （时间步，批量大小，隐藏单元数）
# print(Y.shape, state_new.shape)


# 一个完整的循环神经网络模型
# 注意，rnn_layer只包含隐藏的循环层，我们还需要创建一个单独的输出层
#@save
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            # 输出层
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            # 输出层
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

"""训练与预测"""

# device = d2l.try_gpu()
# net = RNNModel(rnn_layer, vocab_size=len(vocab))
# net = net.to(device)
# 训练模型之前，看看预测
# print(predict_ch8('time traveller', 10, net, vocab, device))

# 使用高级API训练模型
# num_epochs, lr = 500, 1
# train_ch8(net, train_iter, vocab, lr, num_epochs, device)

# d2l.plt.show()