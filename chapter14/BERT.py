import torch
from torch import nn
from d2l import torch as d2l
from chapter10.transformer import EncoderBlock

"""输入表示"""

#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B (+2 为这两个特殊的符号)
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    # 返回BERT输入序列的标记及其相应的片段索引
    return tokens, segments

#@save
class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        # 输入为 2
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            # transformer 的 encoder-block
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        """
        :param tokens: BERT输入序列的标记
        :param segments: 及其相应的片段索引
        :param valid_lens:
        """
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

# 假设词表大小为10000, 演示BERTEncoder的前向推断
# vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
# norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
# encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
#                       ffn_num_hiddens, num_heads, num_layers, dropout)

# 将tokens定义为长度为8的2个输入序列，其中每个词元是词表的索引
# tokens = torch.randint(0, vocab_size, (2, 8))
# segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
# 返回编码结果，其中每个词元由向量表示，其长度由超参数num_hiddens定义
# encoded_X = encoder(tokens, segments, None)
# (batch_size, num_steps, num_hiddens)
# print(encoded_X.shape)  # torch.Size([2, 8, 768])

"""预训练任务"""

# Masked Language Modeling
#@save
class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        # 一个单隐藏层的mlp
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        """
        :param X: 输入 （所有句子BERTEncoder的输出）
        :param pred_positions: 哪些地方要预测 （即哪些词要加 mask）
        :return:
        """
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        # 对每个 masked 的位置都要预测
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


# 演示MaskLM的前向推断
# mlm = MaskLM(vocab_size, num_hiddens)
# 预测第0序列的1、5、2，第1序列的6、1、5
# mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
# 返回encoded_X的所有掩蔽位置mlm_positions处的预测结果mlm_Y_hat
# mlm_Y_hat = mlm(encoded_X, mlm_positions)
# 对于每个预测，结果的大小等于词表的大小
# print(mlm_Y_hat.shape)


# 计算在BERT预训练中的遮蔽语言模型任务的交叉熵损失
# mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
# loss = nn.CrossEntropyLoss(reduction='none')
# mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
# 总共 6 个样本
# print(mlm_l.shape)



# 下一句预测（Next Sentence Prediction）
#@save
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        # 是个 单分类问题
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)

# NextSentencePred实例的前向推断返回每个BERT输入序列的二分类预测
# （正常这里应该要是 encoded_X[:,0,:]，把<cls>抽出来放进）
# encoded_X = torch.flatten(encoded_X, start_dim=1)
# NSP的输入形状:(batchsize，num_hiddens)
# nsp = NextSentencePred(encoded_X.shape[-1])
# nsp_Y_hat = nsp(encoded_X)
# print(nsp_Y_hat.shape)

# 计算两个二元分类的交叉熵损失
# nsp_y = torch.tensor([0, 1])
# nsp_l = loss(nsp_Y_hat, nsp_y)
# print(nsp_l.shape)

"""整合代码"""
# 最终的损失函数是掩蔽语言模型损失函数和下一句预测损失函数的线性组合
#@save
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        # hidden layer，是给nsp用的
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:  # 如果要预测位置
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # encoded_X的形状：（批量大小，最大序列长度，num_hiddens）
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        # 将它抽出来，做二分类问题
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        # 返回编码后的BERT表示、掩蔽语言模型预测 和 下一句预测
        return encoded_X, mlm_Y_hat, nsp_Y_hat










