# %matplotlib inline
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

"""各种图像增广方式"""
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
# d2l.plt.imshow(img)


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    """
    在输入图像img上多次运行图像增广方法aug并显示所有结果
    :param img: 图片
    :param aug: 图片增广的办法
    :param num_rows: 生成几行
    :param num_cols: 生成几列
    :param scale: 画多大
    """
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

# 水平方向随机进行翻转
# apply(img, torchvision.transforms.RandomHorizontalFlip())
# 垂直方向随机进行翻转
# apply(img, torchvision.transforms.RandomVerticalFlip())

# 随机裁剪一个面积为原始面积10%到100%的区域，该区域的宽高比从0.5～2之间随机取值。
# 然后，区域的宽度和高度都被缩放到200像素
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)

# 随机更改图像的亮度
# apply(img, torchvision.transforms.ColorJitter(
#     brightness=0.5, contrast=0, saturation=0, hue=0))
# apply(img, torchvision.transforms.ColorJitter(
#     brightness=0, contrast=0, saturation=0, hue=0.5))
# 创建一个RandomColorJitter实例，并设置如何同时随机更改图像
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)

# 结合多种图像增广方法
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
# apply(img, augs)

"""使用图像增广进行训练"""
# 使用CIFAR-10数据集 (取样自imagenet，宽高更小)
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
# d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

# 简单的变换
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])
# 测试不做变换，直接totensor
test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

def load_cifar10(is_train, augs, batch_size):
    """辅助函数，便于读取图像和应用图像增广"""
    # 在dataset创建的时候，直接transform做增广
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

"""多GPU训练"""
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """用多GPU进行小批量训练"""
    if isinstance(X, list):  # 如果是个list
        # 微调BERT中所需
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """用多GPU进行模型训练"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    """初始化"""
    if type(m) in [nn.Linear, nn.Conv2d]:
        # 其实默认也做了
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    """使用图像增广来训练模型"""
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)


train_with_data_aug(train_augs, test_augs, net)

d2l.plt.show()