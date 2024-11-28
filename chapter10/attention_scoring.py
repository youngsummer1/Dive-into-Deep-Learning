import math
import torch
from torch import nn
from d2l import torch as d2l

"""掩蔽softmax操作"""
#@save
def masked_softmax(X, valid_lens):
    """
    通过在最后一个轴上掩蔽元素来执行softmax操作,设成很小的值使exp后接近0
        X： (batch_size, seq_len, num_classes)
    """
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:  # 若为 1D
            # 每一行都repeat成相同的值，即 X 的 一个样本内所有句子 都按照这样划
            # 形状应该是 (len(valid_lens),shape[1])
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:  # 若为 2D
            # 因为下面会把 X 的前两个轴拉平，再传进去
            # 所以这里 2D 的valid_lens也拉平
            valid_lens = valid_lens.reshape(-1)
        # 前两个轴拉平了
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        # 在最后一个维度（通常是类别维度）进行 softmax 计算
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# 演示此函数是如何工作
# 超出有效长度的值都被掩蔽为0
# (batch_size, num_step, num_classes)
# 此处 valid_lens 代表两个样本的有效长度分别为 2 、3
# print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))

# 使用二维张量，为矩阵样本中的每一行指定有效长度
# print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

"""加性注意力"""

#@save
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # 三个可学习参数，三个线性层
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        # 可以做点正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """
        :param valid_lens: 有多少 键-值对 是需要的 （长度等于query的长度）
                        （对每个query，要考虑多少 键-值对）
        """
        queries, keys = self.W_q(queries), self.W_k(keys)  # 即 Wq*q 和 Wk*k
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        # 出来的结果是 （batch_size，查询的个数，“键－值”对的个数，num_hidden）？把每个query和key都加起来了
        # （能想清楚，但说不出来，好怪）
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)

        # scores = self.w_v(features)
        # print(scores)
        # scores = scores.squeeze(-1)
        # print(scores)

        # 过滤掉不需要的scores
        self.attention_weights = masked_softmax(scores, valid_lens)
        # print('attention_weights:',self.attention_weights)

        # 对 attention_weights 做dropout，即对哪些“键-值”对不需要看了
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        # 最终结果是 (batch_size,查询的个数,值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 演示上面的AdditiveAttention类
# 形状 （批量大小，步数或词元序列长度，特征大小）
# queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values的小批量，两个值矩阵是相同的
# values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
#     2, 1, 1)
# valid_lens = torch.tensor([2, 6])
#
# attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
#                               dropout=0.1)
# attention.eval()
# 注意力汇聚输出的形状为（批量大小，查询的步数，值的维度）
# print(attention(queries, keys, values, valid_lens))

# 尽管加性注意力包含了可学习的参数，但由于本例子中每个键都是相同的， 所以注意力权重是均匀的，由指定的有效长度决定
# d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
#                   xlabel='Keys', ylabel='Queries')

"""缩放点积注意力"""

#@save
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        # ?这个注释是不是忘记删了，这里应该是transpose来交换维度，为了做矩阵乘法的吧
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 演示上述的DotProductAttention类
# queries = torch.normal(0, 1, (2, 1, 2))
# attention = DotProductAttention(dropout=0.5)
# attention.eval()
# print(attention(queries, keys, values, valid_lens))

# 与加性注意力演示相同，由于键包含的是相同的元素， 而这些元素无法通过任何查询进行区分，因此获得了均匀的注意力权重
# d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
#                   xlabel='Keys', ylabel='Queries')




















d2l.plt.show()