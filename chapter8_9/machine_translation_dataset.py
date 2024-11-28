import os
import torch
from d2l import torch as d2l
from text_preprocessing import Vocab

"""下载和预处理数据集"""
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

# 数据集中的每一行都是制表符分隔的文本序列对， 序列对由英文文本序列和翻译后的法语文本序列组成
# 请注意，每个文本序列可以是一个句子， 也可以是包含多个句子的一个段落
#@save
def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

# raw_text = read_data_nmt()
# print(raw_text[:75])

# 几个预处理步骤
#@save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

# text = preprocess_nmt(raw_text)
# print(text[:80])


"""词元化"""
# 在机器翻译中，更喜欢单词级词元化
#@save
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        # 分开英语 和 法语
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    # 英语 ，法语
    return source, target

# source, target = tokenize_nmt(text)
# print(source[:6], target[:6])


# 绘制每个文本序列所包含的词元数量的直方图
# 这个简单的“英－法”数据集中，大多数文本序列的词元数量少于 20 个
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

# show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
#                         'count', source, target);

"""词表"""
# 分别为源语言和目标语言构建两个词表
# src_vocab = Vocab(source, min_freq=2,
#                       reserved_tokens=['<pad>', '<bos>', '<eos>'])
# print(len(src_vocab))


"""加载数据集"""
# 序列样本都有一个固定的长度
#@save
def truncate_pad(line, num_steps, padding_token):
    """
    截断或填充文本序列
        num_steps: 如果句子大于 num_steps ，就把多的截断
        padding_token：如果句子小于 num_steps，就用这个填充
    """
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

# print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

#  转换成小批量数据集用于训练
#@save
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]  # 每个句子结束 加截止符
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)  # 在 dim=1 上相加
    # valid_len -- 这个句子的实际长度 （去除填充）
    return array, valid_len

# print(build_array_nmt(source, src_vocab, 10))

"""训练模型"""

#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """
    返回翻译数据集的迭代器和词表
        num_steps: 每个句子长度
    """
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

# 读出“英语－法语”数据集中的第一个小批量数据
# train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
# for X, X_valid_len, Y, Y_valid_len in train_iter:
#     print('X:', X.type(torch.int32))
#     print('X的有效长度:', X_valid_len)
#     print('Y:', Y.type(torch.int32))
#     print('Y的有效长度:', Y_valid_len)
#     break



d2l.plt.show()