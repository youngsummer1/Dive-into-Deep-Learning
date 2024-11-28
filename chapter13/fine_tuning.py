# %matplotlib inline
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 加载数据集
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))


# 显示
# hotdogs = [train_imgs[i][0] for i in range(8)]
# not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
# d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);

# 使用RGB通道的均值和标准差，以标准化每个通道
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

"""数据增广"""
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

"""定义和初始化模型"""
# pretrained_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
# 输出层
# print(pretrained_net.fc)

finetune_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
# 改输出层（输出改成2）
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
# 只对最后的输出层做初始化
nn.init.xavier_uniform_(finetune_net.fc.weight);

"""训练"""
# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    """
    :param param_group: 若True，则输出层中的模型参数将使用十倍的学习率
    """
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        # 把 非最后一层的参数都拿出来
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        # 最后一层要用 10 倍 的学习率，让它训练的更快
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)

# 训练
# 因为是微调，所以学习率用的很小
# train_fine_tuning(finetune_net, 5e-5)

# 对比 —— 不用finetune时
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)

d2l.plt.show()