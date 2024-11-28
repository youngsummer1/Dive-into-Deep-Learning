# %matplotlib inline
import os
import torch
import torchvision
from d2l import torch as d2l

#@save
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')


"""将所有输入的图像和标签读入内存"""
#@save
def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        # 读原始图片
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        # 每个像素的label，存成png（不会压缩，jpg可能压缩?）
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

# train_features, train_labels = read_voc_images(voc_dir, True)

# 绘制前5个输入图像及其标签
n = 5
# imgs = train_features[0:n] + train_labels[0:n]
# imgs = [img.permute(1,2,0) for img in imgs]
# d2l.show_images(imgs, 2, n);

# 列举RGB颜色值和类名(一一对应)
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 为方便地查找标签中每个像素的类索引，做了下面两个函数
#@save
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        # rgb直接换算成整数， 256进制？
        # 用python自己的话速度很慢
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    # 将通道数丢到最后
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
# 例如，在第一张样本图像中，飞机头部区域的类别索引为1，而背景索引为0。
# y = voc_label_indices(train_labels[0], voc_colormap2label())
# print(y[105:115, 130:140], VOC_CLASSES[1])

"""预处理数据"""
# 使用图像增广中的随机裁剪，裁剪输入图像和标签的相同区域
#@save
def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像 (两个东西要做一样的裁剪才行)"""
    # 为了做成batch要将不同尺寸的图片裁剪成相同大小，对于图片和label的裁剪是相同的

    # get_params --> 返回bounding box
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    # 两者做 一样的裁剪
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
# 演示 （对同一张图 做五次裁剪）
# imgs = []
# for _ in range(n):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
#
# imgs = [img.permute(1, 2, 0) for img in imgs]
# d2l.show_images(imgs[::2] + imgs[1::2], 2, n);

"""自定义语义分割数据集类"""
# 定义 dataset
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        """
        :param is_train:
        :param crop_size: 剪裁大小 （因为要让 让图片大小相同）
        :param voc_dir:
        """
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        """转成比例 [0,1]"""
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        """过滤器，挑出大小 大于 剪裁大小的图片"""
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        """每次返回要做的"""
        # 每张图都要做一次剪裁
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

"""读取数据集"""
# 剪裁大小 （数据集中大部分比这个大，小部分差不多，这里给取了个整）
crop_size = (320, 480)
# dataset
voc_train = VOCSegDataset(True, crop_size, voc_dir)  # read 1114 examples
voc_test = VOCSegDataset(False, crop_size, voc_dir)  # read 1078 examples

# 批量大小
batch_size = 64
# 这里的num_workers 好像不能用 d2l.get_dataloader_workers()，电脑跑不动，直接用0算了
# if __name__ == '__main__':
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                         drop_last=True,
                                         num_workers=0)

for X, Y in train_iter:
    print(X.shape)  # torch.Size([64, 3, 320, 480])
    print(Y.shape)  # torch.Size([64, 320, 480])
    break
"""整合所有组件"""
#@save
def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    # 返回训练集和测试集的数据迭代器
    return train_iter, test_iter

d2l.plt.show()