import pandas as pd
import numpy as np
from torch.utils import data
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from PIL import Image
from tqdm import tqdm
import torchvision
import torchvision.models as models
# 导入数据集
labels_dataFrame = pd.read_csv('../data/Classify-Leaves/train.csv')
# test_dataFrame = pd.read_csv('../data/Classify-Leaves/test.csv')

# print(labels_dataFrame.head())
# print(labels_dataFrame.shape)

# 观察
# print(labels_dataFrame.describe())

"""数据预处理"""
# 把label标签按字母排个序,set()用于去重
leaves_labels = sorted(list(set(labels_dataFrame['label'])))
n_class = len(leaves_labels)
# print(len(leaves_labels))
# print(leaves_labels)
# print(n_class)

# 创建字典 label:num
class_to_num = dict(zip(leaves_labels, range(n_class)))
# print(class_to_num)
# 创建字典 num:label
# num_to_class = dict(zip(range(n_class),leaves_labels))
num_to_class = {v: k for k, v in class_to_num.items()}
# print(num_to_class)
# print(num_to_class == num_to_class2)

"""数据集生成"""
# Dataset (将用于Dataloader)
class LeaveData(data.Dataset):
    def __init__(self, csv_path, file_path, mode = 'train', valid_ratio = 0.2,):
                 # resize_height = 256, resize_width = 256):
        self.file_path = file_path  # 文件根路径
        # self.resize_height = resize_height  # 高
        # self.resize_width = resize_width  # 宽
        self.mode = mode  # 模式
        # 仅在主进程中加载数据
        # if __name__ == '__main__':
        # 读取csv
        self.data_info = pd.read_csv(csv_path)
        # 长度
        self.data_len = len(self.data_info.index)  # 数据集的大小
        self.train_len = int(self.data_len * (1 - valid_ratio))  # 训练数据集的大小

        if mode == 'train':
            self.train_image = np.asarray(self.data_info.iloc[:self.train_len, 0])
            self.train_label = np.asarray(self.data_info.iloc[:self.train_len, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[:, 0])
            self.image_arr = self.test_image
        self.real_len = len(self.image_arr)  # 数据集长度
        print('Finished reading the {} set of Leaves Dataset ({} samples found)'.format(mode, self.real_len))

    def __getitem__(self, index):
        """Dataset必要的函数"""
        # 文件名
        single_image_name = self.image_arr[index]
        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)
        # print(img_as_img)

        # 图像处理
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # if self.mode == 'train':
        #     transform = torchvision.transforms.Compose([
        #         torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),  # 从图片的中心点剪裁出24*24的图片
        #                                                  ratio=(3.0 / 4.0, 4.0 / 3.0)),
        #         torchvision.transforms.RandomHorizontalFlip(),  # 左右翻转
        #         torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
        #                                            saturation=0.4),
        #         # 加入随机噪音
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize([0.485, 0.456, 0.406],  # 对图片的每个通道做均值和方差
        #                                          [0.229, 0.224, 0.225])
        #     ])
        # else:
        #     # valid 和 test 不做数据增强
        #     transform = torchvision.transforms.Compose([
        #         torchvision.transforms.Resize(256),
        #         torchvision.transforms.CenterCrop(224),
        #         # 加入随机噪音
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize([0.485, 0.456, 0.406],  # 依然 对图片的每个通道做均值和方差
        #                                          [0.229, 0.224, 0.225])
        #     ])

        img_as_img = transform(img_as_img)
        # print(img_as_img)
        if self.mode == 'test':
            return img_as_img
        else:
            label = self.label_arr[index]
            # num label
            number_label = class_to_num[label]
            # 返回每一个index对应的图片数据和对应的label
            return img_as_img, number_label
    def __len__(self):
        return self.real_len


train_path = '../data/Classify-Leaves/train.csv'
test_path = '../data/Classify-Leaves/test.csv'
img_path = '../data/Classify-Leaves/'
train_dataset = LeaveData(train_path, img_path, mode='train')
val_dataset = LeaveData(train_path, img_path, mode='valid')
test_dataset = LeaveData(test_path, img_path, mode='test')
# print(train_dataset)
# print(val_dataset)
# print(test_dataset)


# DataLoader
train_loader = data.DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle = False,
    # num_workers = 4
)
val_loader = data.DataLoader(
    dataset = val_dataset,
    batch_size = 32,
    shuffle = False,
    # num_workers = 4
)
test_loader = data.DataLoader(
    dataset = test_dataset,
    batch_size = 128,
    shuffle = False,
    # num_workers = 4
)


"""定义网络"""
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


b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
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
                    nn.Flatten(), nn.Linear(512, 176))
# print(net)

"""试试别人的"""
# 冻结模型前面的层
# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         model = model
#         for param in model.parameters():
#             param.requires_grad = False
# # 调用ResNet并保留原模型参数
# def res_model(num_classes, feature_extract = False, use_pretrained = True):
#     model_ft = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
#     set_parameter_requires_grad(model_ft, feature_extract)
#     num_ftrs = model_ft.fc.in_features
#     model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
#     return model_ft



"""定义超参数"""
# lr, num_epochs, wd = 0.01, 10, 0.01
lr, num_epochs, wd = 0.01, 20, 1e-3

device = d2l.try_gpu()
model_path = './pre_res_model.ckpt'


# net = res_model(176)# 176种叶子
net = net.to(device)
net.device = device

"""训练模型"""
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# def train(net, train_loader, val_loader, num_epochs,
#           lr, wd, device):
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

# 不对啊，只在主进程运行的话，num_workers有什么意义
# if __name__ == '__main__':
net.apply(init_weights)
print('training on', device)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
loss = nn.CrossEntropyLoss()
best_acc = 0.0
for epoch in range(num_epochs):
    print(epoch)
    train_loss = []
    train_accs = []
    net.train()
    # 训练
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            # Compute the accuracy for current batch.
            acc = (y_hat.argmax(dim=-1) == y).float().mean()
            print("acc:", acc)
            # Record the loss and accuracy.
            train_loss.append(l.item())
            train_accs.append(acc)
            pbar.update(1)
    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # 评估
    # net.eval()
    # valid_loss = []
    # valid_accs = []
    # for imgs, labels in tqdm(val_loader):
    #     with torch.no_grad():
    #         logits = net(imgs.to(device))
    #     l = loss(logits, labels.to(device))
    #     # Compute the accuracy for current batch.
    #     acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    #     # Record the loss and accuracy.
    #     valid_loss.append(l.item())
    #     valid_accs.append(acc)
    # # The average loss and accuracy of the training set is the average of the recorded values.
    # valid_loss = sum(valid_loss) / len(valid_loss)
    # valid_acc = sum(valid_accs) / len(valid_accs)

    valid_acc = evaluate_accuracy_gpu(net, val_loader, device)
    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] acc = {valid_acc:.5f}")
    # if the model improves, save a checkpoint at this epoch
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(net.state_dict(), model_path)
        print('saving model with acc {:.3f}'.format(best_acc))

"""模型评估"""
# best_net = nn.Sequential(b1, b2, b3, b4, b5,
#                          nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均汇聚层
#                          nn.Flatten(), nn.Linear(512, 176))
# # best_net = res_model(176)
# best_net = best_net.to(device)
# best_net.load_state_dict(torch.load(model_path))
# best_net.eval()
# """预测"""
# saveFileName = './submission.csv'
# predictions = []
#
# for batch in tqdm(test_loader):
#     imgs = batch
#     with torch.no_grad():
#         imgs_hat = best_net(imgs.to(device))
#     predictions.extend(imgs_hat.argmax(dim=-1).cpu().numpy().tolist())
#
# preds = []
# for i in predictions:
#     preds.append(num_to_class[i])
#
# test_data = pd.read_csv(test_path)
# test_data['label'] = pd.Series(preds)
# submission = pd.concat([test_data['image'], test_data['label']], axis=1)
# submission.to_csv(saveFileName, index=False)
print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!")

