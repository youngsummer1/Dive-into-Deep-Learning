# %matplotlib inline
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from language_model import load_data_time_machine
# 批量大小， 时间步
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

"""独热编码"""
# print(F.one_hot(torch.tensor([0, 2]), len(vocab)))

# 小批量数据形状是二维张量： （批量大小，时间步数）
# X = torch.arange(10).reshape((2, 5))
# 转变形状为 （时间步数，批量大小，词表大小）
# 更方便地通过最外层的维度 （同一个时间步，即 Xt 被放到了一起，方便进行迭代）
# print(F.one_hot(X.T, 28).shape)
# print(F.one_hot(X.T, 28))

"""初始化模型参数"""


def get_params(vocab_size, num_hiddens, device):
    """
    初始化循环神经网络模型的模型参数
    :param vocab_size: 词表大小
    :param num_hiddens:  隐藏单元数
    :return:
    """
    # 输入 和 输出 来自相同的词表， 所以 具有相同的维度
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        # 返回均值 0 ， 方差 0.01 的 均匀分布？
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))  # x->h（输入大小，隐藏单元数）
    W_hh = normal((num_hiddens, num_hiddens))  # h->h
    b_h = torch.zeros(num_hiddens, device=device)  # 偏移
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))  # 隐藏变量 -> 输出 的W
    b_q = torch.zeros(num_outputs, device=device)  # 偏移
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        # 要算梯度，进行更新
        param.requires_grad_(True)
    return params

"""循环神经网络模型"""
def init_rnn_state(batch_size, num_hiddens, device):
    """在初始化时返回隐状态  （因为在0时刻没有上一时刻的隐状态）"""
    # （批量大小，隐藏单元数）
    # 返回一个元组，更容易地处理遇到隐状态包含多个变量的情况
    return (torch.zeros((batch_size, num_hiddens), device=device), )
# state = init_rnn_state(5,3,'cpu')
# print(state)
# print(type(state))




def rnn(inputs, state, params):
    """
    定义了如何在一个时间步内计算隐状态和输出
    :param inputs: x0 ~ xt 所有的时间步输入，形状：(时间步数量，批量大小，词表大小)
    :param state: 初始的隐藏状态
    :param params: 参数
    :return: 所有时刻的输出在垂直方向拼接的矩阵，（当前的隐藏状态，）
    """
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    # 这里 state 是个 tuple ，所以加个 逗号
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:  # 每次做一个时间步的数据
        # 更新 每一步的 H 和 Y
        # 右边的 H 是 前一个时间的隐藏状态
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # 当前时刻对下一个时刻的预测
        # （批量大小，隐藏单元数）*（隐藏单元数，输出数）
        Y = torch.mm(H, W_hq) + b_q  # 注意：是个矩阵
        # 存储所有时刻的输出
        outputs.append(Y)
    # outputs 内的一个个矩阵 在垂直方向上拼接，高度 = 批量大小 * 时间的长度
    # outputs输出形状是（时间步数 * 批量大小，词表大小）
    return torch.cat(outputs, dim=0), (H,)

# 创建一个类来包装这些函数， 并存储从零开始实现的循环神经网络模型的参数
class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        """
        :param vocab_size: 词表大小
        :param num_hiddens: 隐藏层大小
        :param device: 设备
        :param get_params: 获取参数的函数
        :param init_state: 初始隐藏状态
        :param forward_fn:  更新的法则，此处为rnn
        """
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """
        :param X:  （批量大小，时间步数）
        :param state: 隐藏状态
        """
        # X 转置做onehot （做完是个整形，转换成浮点型）
        # 转变形状为 （时间步数，批量大小，词表大小）
        # 更方便地通过最外层的维度 （同一个时间步，即 Xt 被放到了一起，方便进行迭代）
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """定义初始隐藏状态"""
        return self.init_state(batch_size, self.num_hiddens, device)

# 检查输出是否具有正确的形状
# num_hiddens = 512
# net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
#                       init_rnn_state, rnn)
# state = net.begin_state(X.shape[0], d2l.try_gpu())
# Y, new_state = net(X.to(d2l.try_gpu()), state)
# 2 * 5 = 10, 词表大小 - 28， （new_state，） -- （（批量大小，隐藏单元数），）
# print(Y.shape, len(new_state), new_state[0].shape)  # torch.Size([10, 28]) 1 torch.Size([2, 512])


"""预测"""
# 首先定义预测函数来生成prefix之后的新字符
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """
    在prefix后面生成新字符
        prefix: 一个用户提供的包含多个字符的字符串
        num_preds: 要生成多少个词/字符
    """
    # 初始隐藏状态
    state = net.begin_state(batch_size=1, device=device)
    # outputs 用来存放 字符 对应的下标
    outputs = [vocab[prefix[0]]]
    # 把最近预测的词 作为下一个时刻的输入
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 预热期：不断地将隐状态传递到下一个时间步，模型会自我更新(如，获得比初始值更好的隐状态)，但是不生成任何输出
    for y in prefix[1:]:  # 预热期
        # 不关心输出，是关心更新的下个状态
        _, state = net(get_input(), state)
        # outputs 存的是 真实的prefix，而不是预测
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        # 做分类，把预测存入 outputs
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    # 把 index 转成对应的token，然后输出
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# 测试 （鉴于还没有训练网络，它会生成荒谬的预测结果）
# print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))

"""梯度剪裁"""

def grad_clipping(net, theta):  #@save
    """
    裁剪梯度
    （rnn 的 层数会很多，所以得控制梯度爆炸）
    """
    # 取出参数
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # 对所有层平方求和 再 开根号，即求L2norm
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    # 如果 大于 θ 就限制
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

"""训练"""
# 训练一个 epoch
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            # (随机抽样时 状态 也初始化的原因：前后不是连续的，状态不能延续)
           state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 只关心现在开始的计算，之前跟计算图相关的都清除
            # （因为是顺序分区，所以state是从上一句来的）
            # 别人的解释：
            # 由于隐状态计算依赖于先前批量数据，反复累计会使梯度计算变得复杂
            # 为了降低计算量，在每轮训练开始前，先把隐状态先前带有的梯度分离 只专注于该轮的梯度计
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        # Y 的形状（批量大小，时间步数）
        # 转换成（时间步数 * 批量大小），拉长成向量？
        # 转置后再拉长：注意，相邻的得是同一个时间步的
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        # y_hat的形状（时间步数 * 批量大小，词表大小）
        y_hat, state = net(X, state)
        # 变成了标准的多分类问题 来求损失
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            # 梯度剪裁，超过1就限制
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            # 梯度剪裁，超过1就限制
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    # exp(loss / 批量大小) -- 困惑度
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

# 训练函数
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章） (没用valid_iter，偷了懒，因为有predict可以直接看效果)"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    # 封装一下前面的 predict_ch8 函数
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        # ppl -- 困惑度
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

# 因为我们在数据集中只使用了10000个词元， 所以模型需要更多的迭代周期来更好地收敛
# num_epochs, lr = 500, 1
# train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())


# 检查一下使用随机抽样方法的结果
# net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
#                       init_rnn_state, rnn)
# train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
#           use_random_iter=True)

# d2l.plt.show()
