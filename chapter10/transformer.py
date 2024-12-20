import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from chapter10.multi_head_attention import MultiHeadAttention
from chapter8_9.encoder_decoder import EncoderDecoder

"""基于位置的前馈网络"""
#@save
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        """
        :param X: （批量大小，时间步数或序列长度，隐单元数或特征维度）
                    三维的，但 pytorch 默认将前两维都当做批量，最后一维当做特征维
        """
        # 转换成形状为（批量大小，时间步数，ffn_num_outputs）
        return self.dense2(self.relu(self.dense1(X)))

# 改变张量的最里层维度的尺寸，会改变成基于位置的前馈网络的输出尺寸
# 因为用同一个多层感知机对所有位置上的输入进行变换，所以当所有这些位置的输入相同时，它们的输出也是相同的
# ffn = PositionWiseFFN(4, 4, 8)
# ffn.eval()
# print(ffn(torch.ones((2, 3, 4)))[0])

"""残差连接和层规范化"""
# 对比不同维度的层规范化和批量规范化的效果
# ln = nn.LayerNorm(2)
# bn = nn.BatchNorm1d(2)
# X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# 在训练模式下计算X的均值和方差
# print('layer norm:', ln(X), '\nbatch norm:', bn(X))

# 残差连接和层规范化来实现AddNorm类
#@save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """
        :param Y: X 进入某个东西的输出
        """
        # 暂退法也被作为正则化方法使用
        return self.ln(self.dropout(Y) + X)

# 残差连接要求两个输入的形状相同，以便加法操作后输出张量的形状相同
# add_norm = AddNorm([3, 4], 0.5)
# add_norm.eval()
# print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)


"""编码器"""
#@save
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        """包含两个子层：多头自注意力和基于位置的前馈网络，这两个子层都使用了残差连接和紧随的层规范化"""
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

# Transformer编码器中的任何层都不会改变其输入的形状
# X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
# print(encoder_blk(X, valid_lens).shape)


#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，(让它们大小差不多，防止X过小)（embedding 对一个长为d的，通常会设L2-NORM=1）
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            # 没有用 Sequential 是为了记录每一步的attention weights
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

# 创建一个两层的Transformer编码器
encoder = TransformerEncoder(
    200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
encoder.eval()
# 输出的形状是（批量大小，时间步数目，num_hiddens）
# print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)
"""解码器"""
class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        # 记录 i
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:  # 训练阶段
            key_values = X
        else:  # 预测阶段
            # 要把先前预测的也加到 key - value
            # X 应该是上一次预测（是一个个出来的）
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        # 输入为 ：queries, keys, values, valid_lens
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。  （来自编码器的输出）
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

# 为了便于在“编码器－解码器”注意力中进行缩放点积计算和残差连接中进行加法计算
# 编码器和解码器的特征维度都是num_hiddens
# decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
# decoder_blk.eval()
# X = torch.ones((2, 100, 24))
# state = [encoder_blk(X, valid_lens), valid_lens, [None]]
# print(decoder_blk(X, state)[0].shape)


class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        # 做输出
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重 （用于 可视化）
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

"""训练"""
# num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
# lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
# ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
# key_size, query_size, value_size = 32, 32, 32
# norm_shape = [32]
#
# train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
#
# encoder = TransformerEncoder(
#     len(src_vocab), key_size, query_size, value_size, num_hiddens,
#     norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
#     num_layers, dropout)
# decoder = TransformerDecoder(
#     len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
#     norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
#     num_layers, dropout)
# net = EncoderDecoder(encoder, decoder)
# d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# 将一些英语句子翻译成法语
# engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
# fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
# for eng, fra in zip(engs, fras):
#     translation, dec_attention_weight_seq = d2l.predict_seq2seq(
#         net, eng, src_vocab, tgt_vocab, num_steps, device, True)
#     print(f'{eng} => {translation}, ',
#           f'bleu {d2l.bleu(translation, fra, k=2):.3f}')


# 可视化Transformer的注意力权重
# 编码器自注意力权重的形状为（编码器层数，注意力头数，num_steps或查询的数目，num_steps或“键－值”对的数目）
# enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads,
#     -1, num_steps))
# print(enc_attention_weights.shape)
#
# d2l.show_heatmaps(
#     enc_attention_weights.cpu(), xlabel='Key positions',
#     ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
#     figsize=(7, 3.5))

# 可视化解码器的自注意力权重和“编码器－解码器”的注意力权重
# dec_attention_weights_2d = [head[0].tolist()
#                             for step in dec_attention_weight_seq
#                             for attn in step for blk in attn for head in blk]
# dec_attention_weights_filled = torch.tensor(
#     pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
# dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
# dec_self_attention_weights, dec_inter_attention_weights = \
#     dec_attention_weights.permute(1, 2, 3, 0, 4)
# print(dec_self_attention_weights.shape, dec_inter_attention_weights.shape)

# Plusonetoincludethebeginning-of-sequencetoken
# d2l.show_heatmaps(
#     dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
#     xlabel='Key positions', ylabel='Query positions',
#     titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))


# 输出序列的查询不会与输入序列中填充位置的词元进行注意力计算
# d2l.show_heatmaps(
#     dec_inter_attention_weights, xlabel='Key positions',
#     ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
#     figsize=(7, 3.5))
#
# d2l.plt.show()

