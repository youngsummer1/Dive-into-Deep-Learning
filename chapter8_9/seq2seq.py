import collections
import math
import torch
from torch import nn
from d2l import torch as d2l
from encoder_decoder import EncoderDecoder
"""编码器"""


#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        """
        :param vocab_size: 输入词表的大小
        :param embed_size: 特征向量的维度
        :param num_hiddens: 隐藏单元数
        :param num_layers: 层数
        :param dropout: 暂退
        """
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层 （将输入数据转换为高维向量表示）
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)
        # 注意，没有输出层，只要拿到 rnn 最终的状态就好

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

# 实例化上述编码器的实现
# 两层门控循环单元编码器，其隐藏单元数为 16
# encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
#                          num_layers=2)
# 评估模型 （dropout不生效）
# encoder.eval()
# 小批量的输入序列X（批量大小为 4，时间步为 7）
# X = torch.zeros((4, 7), dtype=torch.long)
# 最后一层的隐状态的输出 output  （时间步数，批量大小，隐藏单元数）
# output, state = encoder(X)
# print(output.shape)
# 最后一个时间步的多层隐状态的形状是 （隐藏层的数量，批量大小，隐藏单元的数量）
# print(state.shape)

"""解码器"""

class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 嵌入层 （将输入数据转换为高维向量表示）
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # embed_size + num_hiddens --> 输入X 和 上下文context 在 dim=2 拼接了，所以得加
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        # 输出层 （输出为 词表大小）
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # enc_outputs --> 是这两个东西 (output, state)
        # 取出 state
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)，再把时间步放在前面
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        # (即在dim=0，重复num_steps次)
        # （state[-1] --> 最后时刻最后一层的输出,形状为 （batch_size,num_hiddens） ）
        # context 的形状 (num_steps,batch_size,num_hiddens)
        context = state[-1].repeat(X.shape[0], 1, 1)

        # 输入 和 上下文context 在 dim=2 拼接（concatenate）
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        # output 进行一个输出， batch_size 换到前面
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

# 实例化解码器
# decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
#                          num_layers=2)
# decoder.eval()
# state = decoder.init_state(encoder(X))
# 输出形状变为（批量大小，时间步数，词表大小）
# output, state = decoder(X, state)
# print(output.shape, state.shape)

"""损失函数"""
# 通过零值化屏蔽不相关的项
#@save
def sequence_mask(X, valid_len, value=0):
    """
    在序列中屏蔽不相关的项
        X: 输入   ,形状为 (batch_size, seq_len)
        valid_len：有效长度
        value：屏蔽值，如 valid_len 之外的都变成 0
    """
    maxlen = X.size(1)
    # [None, :] --> 将 arange 的形状扩展为 (1, maxlen)
    # [:, None] --> 将 valid_len 的形状扩展为 (batch_size, 1)
    # mask, 形状为 (batch_size, maxlen)
    # 第 i 行中，如果 j < valid_len[i] 则mask[i, j] 为 True ；否则为 False
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # 取反 mask ，将所有无效部分的位置替换为指定的值 value
    X[~mask] = value
    return X

# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(sequence_mask(X, torch.tensor([1, 2])))
# X = torch.ones(2, 3, 4)
# print(sequence_mask(X, torch.tensor([1, 2]), value=-1))


# 扩展softmax交叉熵损失函数来遮蔽不相关的预测
#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    带遮蔽的softmax交叉熵损失函数
    （填充的不参加 softmax）
    """
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        # 与填充词元对应的掩码将被设置为0
        weights = sequence_mask(weights, valid_len)
        # 不进行求和
        self.reduction='none'
        # num_step放在后面是因为这个函数是继承nn.crossentropy，
        # 后者要求的输入类型（mini-batch，类别，维度)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        # weight 为 1 的才留下来
        # 对每个句子取平均，每个样本返回一个loss
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

# 代码健全性检查
# loss = MaskedSoftmaxCELoss()
# 3 -- 批量大小 ，4 -- num_steps  ，10 --- vocab_size
# 有效长度为 4、2、0，所以第一个序列的损失应为第二个序列的两倍，而第三个序列的损失应为零
# print(loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
#      torch.tensor([4, 2, 0])))

"""训练"""

#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # begin of sentence  -- bos
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            # Y 去除最后一个，在最前面加上 bos
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            # 不用传Y_valid_len，因为目前没用到，算loss的时候用
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            # 梯度剪裁
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')

# 创建和训练一个循环神经网络“编码器－解码器”模型
# embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
# batch_size, num_steps = 64, 10
# lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
#
# train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
# encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
#                         dropout)
# decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
#                         dropout)
# net = EncoderDecoder(encoder, decoder)
# train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

"""预测"""
# 预测时，每个解码器当前时间步的输入都将来自于前一时间步的预测词元
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴 (就是把bos包成tensor，前面再加个维度)
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)  # dim=2 就是最后一维，是对各个可能性预测的概率
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()  # 把dim=0维去掉，再....
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

"""预测序列的评估"""
# BLEU的代码实现
def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        # 匹配的数量，用于查询的label
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:  # 如果dict中该序列的数量大于 0
                num_matches += 1
                # dict中该序列的数量 - 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

# 将几个英语句子翻译成法语，并计算BLEU的最终结果
# engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
# fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
# for eng, fra in zip(engs, fras):
#     translation, attention_weight_seq = predict_seq2seq(
#         net, eng, src_vocab, tgt_vocab, num_steps, device)
#     print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')



# d2l.plt.show()
















