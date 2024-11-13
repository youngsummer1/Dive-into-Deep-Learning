import torch
from torch import nn
from d2l import torch as d2l

"""VGG块"""
def vgg_block(num_convs, in_channels, out_channels):
    """
    :param num_convs: 多少个卷积层
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    """
    layers = []
    # 添加卷积层 、 激活函数
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    # 添加 池化层
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    # * 解包
    return nn.Sequential(*layers)

"""VGG网络"""
# 5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层 (VGG-11)
# 第一个模块有64个输出通道，每个后续模块将输出通道数量翻倍
# 因为每块都有个池化层会让宽高/2, 224 / 32 = 7
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    """构建VGG网络"""
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),  # * 解包，展开
        # 全连接层部分 （224 / 32 = 7）
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

# net = vgg(conv_arch)
#
# X = torch.randn(size=(1, 1, 224, 224))
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__,'output shape:\t',X.shape)

"""训练"""
ratio = 4
# 因为太大了，所以给每个通道数除4，好训练
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

d2l.plt.show()