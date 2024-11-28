import collections
import re
from d2l import torch as d2l

"""读取数据集"""
# 读取 科幻小说 —— 《时间机器》
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        # 逐行读取
        lines = f.readlines()
    # 把 不是大小写字母的 全变成 空格 （让数据集更简单）（最简单的预处理）
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# lines = read_time_machine()
# print(f'# 文本总行数: {len(lines)}')
# print(lines[0])
# print(lines[10])

"""词元化"""
# 这里word表示单词和字符串比如good和iiiiii都是word，char表示单个字母比如a
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':  # 一整个词
        return [line.split() for line in lines]
    elif token == 'char':  # 字符
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

# tokens = tokenize(lines)
# for i in range(11):
#     print(tokens[i])

"""词表"""
class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        :param tokens: 词元列表
        :param min_freq: 最少出现次数，若小于这个数，就丢掉
        :param reserved_tokens:
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],  # (token, frequency)，x[1] 表示按频率（frequency）进行排序
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """返回 tokens 对应的 indexes"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """返回 indexes 对应的 tokens"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    # 统计可迭代对象中元素出现的次数
    # print(collections.Counter(tokens))

    return collections.Counter(tokens)

# 构建词表
# vocab = Vocab(tokens)
# print(list(vocab.token_to_idx.items())[:10])

# 将每一条文本行转换成一个数字索引列表
# for i in [0, 10]:
#     print('文本:', tokens[i])
#     print('索引:', vocab[tokens[i]])


"""整合所有功能"""
# 将所有功能打包到load_corpus_time_machine函数中
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    # 为了简化后面章节中的训练，使用字符（而不是单词）实现文本词元化
    tokens = tokenize(lines, 'char')
    # 返回的是对应的token
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    # (列表是一大串数字)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# corpus, vocab = load_corpus_time_machine()

# print(len(corpus))
# print(len(vocab))  # 长度 28 （26个字母 + 空格 + unk）
