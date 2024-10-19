import matplotlib.pyplot as plt
import torch
from IPython import display
from d2l import torch as d2l

# 读取数据集fashion_mnist
batch_size = 256
# 封装在Fashion_MNIST_dataset.py中
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""初始化模型参数"""
# 每个图片都是(1, 28, 28), 所以这里要将它拉长成向量 784 = 28 * 28
num_inputs = 784
# 数据集有10个类别，所以网络输出维度为10
num_outputs = 10
# 权重 （正态分布初始化）（行数 --输入个数， 列数 --输出个数）
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 偏移
b = torch.zeros(num_outputs, requires_grad=True)

"""定义softmax操作"""
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# print(X.sum(0, keepdim=True)) # 按维度 0 求和
# tensor([[5., 7., 9.]])
# print(X.sum(1, keepdim=True)) # 按维度 1 求和
# tensor([[ 6.],
#         [15.]])

def softmax(X):
    """softmax操作"""
    # 对每个项求幂 e
    X_exp = torch.exp(X)
    # X 是矩阵，所以要分别对每一行做softmax （每一行应该都是一个样本？）
    # （这里先对每一行进行求和）（n * m -> n * 1）
    partition = X_exp.sum(1, keepdim=True)
    # 广播机制，沿为 1 的那一维推广(n * 1 -> n * m)
    return X_exp / partition  # 这里应用了广播机制

# 测试softmax
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
# print(X_prob)
# tensor([[0.0799, 0.1934, 0.0502, 0.3379, 0.3386],
#         [0.1311, 0.4326, 0.1276, 0.0464, 0.2623]])
# print(X_prob.sum(1, keepdim=True))
# tensor([[1.0000],
#         [1.0000]])

def net(X):
    """定义模型"""
    # torch.matmul() --矩阵乘法
    # X.reshape((-1, W.shape[0])) --即将每张原始图像展平为向量，列数784
    # X --行为批数大小 --列为图像展平大小
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 以下为：
# 怎么在预测值中，根据标号，取出对应的预测值
y = torch.tensor([0, 2]) # 形状 1 * 2
# 假设每个样本有3类
# 这每一行都对应一个样本的3类的预测值
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y_hat[[0,1],[0,2]]，其中[0,1]中的0是指取第一个样本（0.1,0.3,0.6），1是取第二个样本（0.3,0.2,0.5）。[0,2]就是位置，参考Python的列表索引
# 即取出[0, 0] 和 [1, 2]
# print(y_hat[[0, 1], y])
# tensor([0.1000, 0.5000])

def cross_entropy(y_hat, y):
    """
    交叉熵损失函数
        y_hat: 预测值
        y：真实值
    """
    # 这里就是取出y_hat中
    # range(len(y_hat)) --行， y --列 的数据
    # 前后一一对应 （来复习的我看不懂的话，再理解一下上面的那个用法）
    return - torch.log(y_hat[range(len(y_hat)), y])

# 这里取到了处理后的 [0, 0] 和 [1, 2]的数据
# print(cross_entropy(y_hat, y))
# tensor([2.3026, 0.6931])

"""分类精度"""
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    # 行数 > 1 && 列数 > 1 时
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 这里argmax取最大列（最大的预测概率）其实就是取预测的类型结果，下面再和标签比较
        y_hat = y_hat.argmax(axis=1)
    # 转成 y 的数据类型，再作比较
    cmp = y_hat.type(y.dtype) == y
    # 预测正确的数量，转成浮点数
    # (不直接在这里除len(y): 因为是要统计整个数据集预测正确的数量，再去除)
    return float(cmp.type(y.dtype).sum())
# 预测正确的概率
# print(accuracy(y_hat, y) / len(y))
# 0.5

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n # 一个 1 * n的向量？

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
def evaluate_accuracy(net, data_iter):  #@save
    """
    计算在指定数据集上模型的精度
        net：模型
        data_iter：数据迭代器
    """
    # 如果是用 torch.nn 实现的模型
    if isinstance(net, torch.nn.Module):
        # 评估模式：不计算梯度，只做forward()
        # 设了后很多和梯度相关的操作可以不做，可能节省性能？
        net.eval()  # 将模型设置为评估模式
    # 累加器（定义在下面）
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            # 参数1：预测正确的数量， 参数2：样本总数
            metric.add(accuracy(net(X), y), y.numel())
    # 返回模型在数据迭代器上的精度
    return metric[0] / metric[1]



# 测试模型精度
# 发现不在 if __name__ == '__main__': 中执行的话，会报错
# if __name__ == '__main__':
#     # 这里是使用随机权重初始化net模型，所以精度不高
#     print(evaluate_accuracy(net, test_iter))

"""训练"""
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """
    训练模型一个迭代周期（定义见第3章）
        net：模型
        train_iter：
        loss：损失函数
        updater：
    """
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        # 训练模式：要计算梯度
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X) # 预测值
        l = loss(y_hat, y) # 损失函数
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad() # 梯度设为0
            l.mean().backward() # 计算梯度
            updater.step() # 参数更新
        else: # 自己实现的update
            # 使用定制的优化器和损失函数
            l.sum().backward() # 因为是自己实现的update，所以 l 会是一个向量？
            updater(X.shape[0]) # 根据批量大小update
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    # metric[0] --所有loss的累加，metric[1] --分类正确的样本数 ，metric[2] --样本数
    return metric[0] / metric[2], metric[1] / metric[2]

# 一个辅助函数，可以实时看到训练过程变化（没看懂，有缘来看）
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.draw()
        plt.pause(0.01)
        display.display(self.fig)
        display.clear_output(wait=True)
        # d2l.plt.show()

# 训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    # 可视化
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    # 训练 num_epochs 次
    for epoch in range(num_epochs):
        # 训练一次
        # train_metrics --训练误差
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # 在测试数据集test_iter，评估精度
        test_acc = evaluate_accuracy(net, test_iter)
        # 在 animator 中显示
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def updater(batch_size):
    # 包装一下sgd
    return d2l.sgd([W, b], lr, batch_size)

# 训练10个周期
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

"""预测"""
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    # 拿出一个测试样本
    for X, y in test_iter:
        break
    # 真实标号
    trues = d2l.get_fashion_mnist_labels(y)
    # 预测标号
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()
# 测试预测结果
# 发现不在 if __name__ == '__main__': 中执行的话，会报错
# if __name__ == '__main__':
#     # 奇了怪了，我的预测怎么差这么多
#     predict_ch3(net, test_iter)