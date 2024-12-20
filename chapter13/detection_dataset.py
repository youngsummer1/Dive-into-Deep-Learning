# %matplotlib inline
import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

"""下载数据集"""
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

"""读取数据集"""
#@save
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    # 除256是将 像素 存成 0-1 的数
    return images, torch.tensor(targets).unsqueeze(1) / 256


# 创建一个自定义Dataset实例来加载香蕉检测数据集
#@save
class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

# 加载香蕉检测数据集
#@save
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

# 读取一个小批量
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
# print(batch[0].shape, batch[1].shape)  # torch.Size([32, 3, 256, 256]) torch.Size([32, 1, 5])

"""演示"""
# batch[0] --> 图像， batch[1] --> 标签
# 取当前批次的前10张图像，形状为(批量大小, 通道数, H, W)，将其形状改为（批量大小，H，W，通道数）
# 再除255，进行归一化为[0,1]的浮点数
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
# 显示前10张图像，使用2行5列的布局
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    # 且因为该数据集一个图像只有一个框，m=1，所以直接显示一个框就好
    # 乘 edge_size 是因为大小是用 0-1 数来存的
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])


d2l.plt.show()