import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

"""ResNet块"""
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        """
        :param input_channels: 输入通道数
        :param num_channels: 输出通道数
        :param use_1x1conv: 是否使用1 * 1 卷积
        :param strides: 步幅
        """
        super().__init__()
        # 可变高宽
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        # 高宽不变
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        # 如果输入输出通道数不一样，就要用1 * 1 卷积改变
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # 每个bn有自己的参数要学
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# 查看输入和输出形状一致的情况
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6) # 样本数，输出通道数，高，宽
Y = blk(X)
# print(Y.shape)

# 增加输出通道数的同时，减半输出的高和宽的情况
blk = Residual(3,6, use_1x1conv=True, strides=2)
# print(blk(X).shape)

"""ResNet模型"""
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 大残差块
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    """
    :param input_channels: 输入通道数
    :param num_channels: 输出通道数
    :param num_residuals: 要多少个resnet块
    :param first_block: 是否是第一个块
    :return:
    """
    blk = []
    for i in range(num_residuals):
        # first block 指的是整个结构里的第一个 i=0仅仅是这个block里面第一个
        # 如果是first block就不做高宽减半了，因为前面做太多了（如这里的b2?）
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
# 高宽减半，通道数加倍
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),  # 全局平均汇聚层
                    nn.Flatten(), nn.Linear(512, 10))

# 观察输入形状变化
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    # print(layer.__class__.__name__,'output shape:\t', X.shape)

"""训练模型"""
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

d2l.plt.show()