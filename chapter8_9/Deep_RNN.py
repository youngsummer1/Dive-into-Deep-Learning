import torch
from torch import nn
from d2l import torch as d2l
from language_model import load_data_time_machine
from RNN import train_ch8
from RNN_concise import RNNModel

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)



vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
# 通过num_layers的值来设定隐藏层数
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)

"""训练与预测"""
num_epochs, lr = 500, 2
# 由于使用了长短期记忆网络模型来实例化两个层，因此训练速度被大大降低了
train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)


d2l.plt.show()