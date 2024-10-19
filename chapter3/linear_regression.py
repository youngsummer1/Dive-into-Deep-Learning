# 下面这个是将图表嵌入到notebook中，ipython的内置函数，可用d2l.plt.show()
# %matplotlib inline
import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    # normal是高斯分布 matmul是矩阵乘法
    X = torch.normal(0, 1, (num_examples, len(w))) # 0 --均值，1 --标准差, 第三个参数 --形状
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1)) # -1 表示自动计算

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# print('features:', features[0],'\nlabel:', labels[0])
# features: tensor([-0.0821, -1.6111])
# label: tensor([9.5214])

"""用于画图"""
# d2l.set_figsize()
# # detach() -> 在pytorch的一些版本需要先处理，才能转成numpy (现在的plt好像不需要了)
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# # 显示图
# d2l.plt.show()


def data_iter(batch_size, features, labels):
    """
    读取数据集， 使用for的话，每次返回一个batch_size大小的随机样本
        batch_size: 批量大小
        features： 特征
        labels： 标号
    """
    num_examples = len(features) # 取 features 第一维， 即 1000
    indices = list(range(num_examples)) # 下标
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices) # 将下标打乱
    for i in range(0, num_examples, batch_size): # 0 ~ num_examples，每次跳batch_size的大小
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])# 取从 i ~ i + batch_size的数据，min是防止超出样本大小
        # yield python的生成器
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y) # X 为 10*2 的 tensor， y 为 10*1 的tensor
#     break # 只取了一次值

"""初始化模型参数"""
# 都设置requires_grad=True，是为了让派生出的tensor都自动求导
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True) # 权重
b = torch.zeros(1, requires_grad=True) # 偏差 ， 为标量

"""定义模型"""
def linreg(X, w, b):  #@save
    """线性回归模型"""
    # y = X * w + b
    return torch.matmul(X, w) + b

"""定义损失函数"""
def squared_loss(y_hat, y):  #@save
    """
    均方损失
        y_hat: 预测值
        y: 真实值
    """
    # 元素个数一样，但形状不一定； 这里没有求均值
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

"""定义优化算法"""
def sgd(params, lr, batch_size):  #@save
    """
    小批量随机梯度下降
        params： 参数
        lr： 学习率
        batch_size： 批量大小
    """
    # 因为是用来更新参数的，所以不用返回值（所以传的是地址吗？）
    with torch.no_grad(): # 让pytorch在这个上下文不要构建计算图，因为只要参数更新，不用梯度计算，节约内存
        for param in params:
            # squared_loss 没求的均值在这求了(因为是线性关系，所以在上面或下面求都行)
            # batch_size 不是很正确，因为最后一个批量可能会少一些
            param -= lr * param.grad / batch_size
            param.grad.zero_() # 清零梯度

    """训练"""
lr = 0.03 # 学习率
num_epochs = 3 # 整个数据扫三遍
net = linreg # 模型
loss = squared_loss # 损失

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels): # 每次拿出一个批量大小的X,y
        l = loss(net(X, w, b), y)  # X和y的小批量损失(y真实-y实际)
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward() # 求和算梯度
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad(): # 不用算梯度
        train_l = loss(net(features, w, b), labels) # labels 是真实值
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 因为用的是自己的数据集，知道真正的参数，所以比一下就知道了
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')