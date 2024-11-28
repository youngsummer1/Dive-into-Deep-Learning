import random
import torch
import re
from d2l import torch as d2l
from text_preprocessing import load_corpus_time_machine

# 读取 科幻小说 —— 《时间机器》
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        # 逐行读取
        lines = f.readlines()
    # 把 不是大小写字母的 全变成 空格 （让数据集更简单）（最简单的预处理）
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

"""自然语言统计"""
# tokens = d2l.tokenize(read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
# corpus = [token for line in tokens for token in line]
# vocab = d2l.Vocab(corpus)
# 打印前 10 个最常用的（频率最高的）单词
# print(vocab.token_freqs[:10])

# 词频衰减的速度相当地快
# freqs = [freq for token, freq in vocab.token_freqs]
# d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
#          xscale='log', yscale='log')

# 二元语法
# bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]  # 每次是拿到两个
# bigram_vocab = d2l.Vocab(bigram_tokens)
# print(bigram_vocab.token_freqs[:10])

# 三元语法
# trigram_tokens = [triple for triple in zip(
#     corpus[:-2], corpus[1:-1], corpus[2:])]
# trigram_vocab = d2l.Vocab(trigram_tokens)
# print(trigram_vocab.token_freqs[:10])

# 直观地对比三种模型中的词元频率：一元语法、二元语法和三元语法
# bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
# trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
# d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
#          ylabel='frequency: n(x)', xscale='log', yscale='log',
#          legend=['unigram', 'bigram', 'trigram'])


"""随机采样"""
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """
    使用随机抽样生成一个小批量子序列
        corpus： 语料
        batch_size： 每个小批量中子序列样本的数目
        num_steps：每个子序列中预定义的时间步数
    """
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 能生成多少个子序列
    # 减去1，是因为我们需要考虑标签 （要给最后一个序列留一个数字当标签）
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引 （的列表）
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]
    # 能划出几个 batch
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):  # i --> 每个batch的起始
        # 在这里，initial_indices包含子序列的随机起始索引
        # 拿出本个 batch 的所有起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        # X 整体往后推一个数据 的数据
        Y = [data(j + 1) for j in initial_indices_per_batch]
        # yield 会将函数视为一个 `generator`
        yield torch.tensor(X), torch.tensor(Y)

# 验证： 生成一个 0 - 34 的序列
# my_seq = list(range(35))
# 可以生成 [(35 - 1) / 5] / 2 = 3 个小批量
# for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
#     print('X: ', X, '\nY:', Y)

"""顺序分区"""
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size  # 变成可以被 batch_size 整除的样子
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

# for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
#     print('X: ', X, '\nY:', Y)

"""整合"""
# 整合到一个类中
class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """
        :param use_random_iter: 是否使用随机
        :param max_tokens:  限制数据的最大长度
        """
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

# 总和
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

d2l.plt.show()