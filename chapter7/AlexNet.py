import torch
from torch import nn
from d2l import torch as d2l

# 实现AlexNet
net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    # ？？为什么加padding = 1， （224 - 11 + 4 + 2）/ 4 = 54
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),  # 这里输入通道为1，因为使用的是fashionMNIST，如果是imageNet应该是3
    nn.MaxPool2d(kernel_size=3, stride=2),  # (54 - 3) / 2 + 1 = 26
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),  # (26 - 3) / 2 + 1 = 12
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),  # 除第0维，其他维拉伸
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),  # 6400 = 256 * 5 * 5
    nn.Dropout(p=0.5),  # 丢弃层
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))

# 观察每一层输出的形状
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    # print(layer.__class__.__name__,'output shape:\t',X.shape)

"""训练"""
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)  # resize，拉成224 * 224（没什么作用，就是模拟）

lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

d2l.plt.show()