# %matplotlib inline
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

"""构造模型"""
# 使用在ImageNet数据集上预训练的ResNet-18模型来提取图像特征
pretrained_net = torchvision.models.resnet18(weights='DEFAULT' )
# print(list(pretrained_net.children())[-3:])
# 去除最后几层全局平均汇聚层和全连接层
net = nn.Sequential(*list(pretrained_net.children())[:-2])

# 给定高度为320和宽度为480的输入，net的前向传播将输入的高和宽减小至原来的 1 / 32 ，即10和15
X = torch.rand(size=(1, 3, 320, 480))
# print(net(X).shape)  # torch.Size([1, 512, 10, 15])

# 使用 1 * 1 卷积层将输出通道数转换为Pascal VOC2012数据集的类数（21类）
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
# 转置卷积，将特征图的高度和宽度增加32倍
# (h + 64 - 16 * 2 - 32) * 32 = 32 h
# 或者这样算 h * 32 + 64 - 16 * 2 - 32 = 32 h
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))

"""初始化转置卷积层"""

def bilinear_kernel(in_channels, out_channels, kernel_size):
    """双线性插值 的 卷积核"""
    # 插值过程的对称中心位置
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:  # 奇数
        center = factor - 1
    else:  # 偶数
        center = factor - 0.5
    # 创建网格坐标 og[0]、[1] 为 行、列索引
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    # 计算位置到中心的距离，离中心越远，权重越小；越近，权重越大
    # 行和列 权重相乘，生成二维的双线性权重分布
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    # 为每个输入输出通道分配核权重
    weight[range(in_channels), range(out_channels), :, :] = filt
    # 返回包含双线性插值核的权重张量
    return weight

# 构造一个将输入的高和宽放大 2 倍的转置卷积层，并将其卷积核用bilinear_kernel函数初始化
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

# 读取图像X，将上采样的结果记作Y
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)  # 转置卷积这里的核用的是双线性插值的核，因此卷积结果应该是原图放大
out_img = Y[0].permute(1, 2, 0).detach()

# 打印图像
# d2l.set_figsize()
# print('input image shape:', img.permute(1, 2, 0).shape)  #  torch.Size([561, 728, 3])
# d2l.plt.imshow(img.permute(1, 2, 0))
# print('output image shape:', out_img.shape)  # torch.Size([1122, 1456, 3])
# d2l.plt.imshow(out_img)


# 全卷积网络用双线性插值的上采样初始化转置卷积层
# 对于 1 * 1 卷积层，我们使用Xavier初始化参数
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

"""读取数据集"""
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

"""训练"""
def loss(inputs, targets):
    # 因为对 高宽 所有的像素都做预测（变成了个矩阵），所以对矩阵要求均值
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

"""预测"""
def predict(img):
    # 需要将输入图像在各个通道做标准化，并转成卷积神经网络所需要的四维输入格式
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)  # 在通道维上做argmax
    return pred.reshape(pred.shape[1], pred.shape[2])


def label2image(pred):
    # 为了可视化预测的类别给每个像素，我们将预测类别映射回它们在数据集中的标注颜色
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    # 在左上角剪裁
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    # 预测
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
# 截取的区域，预测结果，标注的类别
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);

d2l.plt.show()