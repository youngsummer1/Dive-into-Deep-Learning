import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from language_model import load_data_time_machine
from RNN import RNNModelScratch,train_ch8
from RNN_concise import RNNModel

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

"""初始化模型参数"""
def get_params(vocab_size, num_hiddens, device):
    """实例化与更新门、重置门、候选隐状态和输出层相关的所有权重和偏置"""
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        """初始化三个权重"""
        return (normal((num_inputs, num_hiddens)),  # 输入，隐藏
                normal((num_hiddens, num_hiddens)),  # 隐藏， 隐藏
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

"""定义模型"""
# 定义隐状态的初始化函数
def init_gru_state(batch_size, num_hiddens, device):
    # 返回一个 tuple
    # 第一个的形状为（批量大小，隐藏单元个数）
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 定义门控循环单元模型
def gru(inputs, state, params):
    """
    :param inputs: x0 ~ xt 所有的时间步输入，形状：(时间步数量，批量大小，词表大小)
    """
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        # @ --> 矩阵乘法
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)  # 更新门
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)  # 重置门
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)  # 候选隐藏状态
        H = Z * H + (1 - Z) * H_tilda  # 隐藏状态
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

"""训练与预测"""
# 训练和预测的工作方式与 8.5节完全相同
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
# model = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
#                             init_gru_state, gru)
# train_ch8(model, train_iter, vocab, lr, num_epochs, device)


"""简洁实现"""

num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = RNNModel(gru_layer, len(vocab))
model = model.to(device)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)


d2l.plt.show()