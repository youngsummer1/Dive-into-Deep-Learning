import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import hashlib
import os
import tarfile
import zipfile
import requests

train_data = pd.read_csv('../data/california-house-prices/train.csv')
test_data = pd.read_csv('../data/california-house-prices/test.csv')

# print(train_data.columns)
# print(test_data.columns)
# print(train_data.shape)
# print(test_data.shape)

"""数据预处理"""
# 看一下每列数据的有多少种
# for feature in train_data.columns:
#     print(feature.ljust(20),len(train_data[feature].unique()))


# 去除复数的字符串数据，感觉处理不了
# reduce_cols = ['Id','Address','Summary','Heating','Cooling','Parking','Bedrooms',
#                'Region','Flooring','Heating features','Cooling features','Appliances included',
#                'Laundry features']
# for c in reduce_cols:
#     del train_data[c], test_data[c]

# print(train_data.columns)
# print(test_data.columns)

# 去除soldprice
all_features = pd.concat((train_data.iloc[:, 4:], test_data))
# print(all_features.head())

# print(all_features.dtypes)
# print(all_features['Last Sold On'].head())

# 规整一下时间
all_features['Listed On'] = pd.to_datetime(all_features['Listed On'], format="%Y-%m-%d")
all_features['Last Sold On'] = pd.to_datetime(all_features['Last Sold On'], format="%Y-%m-%d")

#清洗一下Bedrooms列的数据
Bedrooms_col = all_features['Bedrooms']
temp_Bedrooms = []
for item in Bedrooms_col:
    if item=='nan':
        temp_Bedrooms.append(0)
    elif isinstance(item,str):
        n = len(item.split(','))
        if n ==0:
            temp_Bedrooms.append(1)
        else:
            temp_Bedrooms.append(n)
    else:
        temp_Bedrooms.append(item)
all_features['Bedrooms'] = temp_Bedrooms

# 将数值特征缩放到 均值0 方差1， 将缺失值替换为相应特征的平均值（即0）
# 标准化数据
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index # 这里认为不是object的都是数值
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())  # 标准化为 x-μ / σ
)

# print(all_features.isnull().sum())

# 缺失值补为平均值 (即0)
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# print(all_features.isnull().sum())
# print(all_features.head())

# 因为感觉这些字符串都处理不了，one-hot后太大了，所以干脆只取类别较少的字符串和数字
features = list(numeric_features)
features.extend(['Type','Bedrooms'])
all_features = all_features[features]
# print(all_features.head())
# 处理离散值
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features * 1

# print(all_features.shape)

# 从 pandas 格式中提取Numpy格式，并将其转换为张量表示用于训练
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32,device=torch.device('cuda'))
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32,device=torch.device('cuda'))
# 处理一下Listed Price， 缺失值填充Sold Price的值
label_list = []
for item in train_data.iterrows():
    if item[1]['Listed Price'] ==0:
        item= item[1]['Sold Price']
        label_list.append(item)
    else:
        label_list.append(item[1]['Listed Price'])
train_data['Listed Price'] = label_list
for item in train_data['Listed Price']:
    if item ==0:
        print('存在0值')

train_labels = torch.tensor(
    train_data['Sold Price'].values.reshape(-1, 1), dtype=torch.float32,device=torch.device('cuda'))


"""训练"""
# 均方误差，返回所有样本损失的平均值
loss = nn.MSELoss()
in_features = train_features.shape[1]

class Net(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.lin1 = nn.Linear(in_features, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        # 激活函数 ， 使用relu
        self.relu = nn.ReLU()

    def forward(self, X):
        # 第一个隐藏层输出
        H1 = self.relu(self.lin1(X))
        # 只有在训练模型时才使用dropout
        # if self.training == True:
        #     # 在第一个全连接层之后添加一个dropout层
        #     H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        # if self.training == True:
        #     # 在第二个全连接层之后添加一个dropout层
        #     H2 = dropout_layer(H2, dropout2)
        H3 = self.relu(self.lin3(H2))
        # 输出层不做 dropout
        out = self.out(H3)
        return out

def get_net():
    # 单层线性回归
    # net = nn.Sequential(nn.Linear(in_features, 1))

    # MLP
    net = Net(in_features)

    # 单隐藏层试试
    # net = nn.Sequential(nn.Linear(in_features,256), nn.ReLU(), nn.Linear(256,1))
    return net

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    # 把数据范围限制在 1 ~ infinite
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    # 根号loss(log(y_hat)-log(y))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    """训练函数"""
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 使用 Adam 优化算法
    # Adam算法是自适应矩估计算法，是目前最先进的梯度下降算法，梯度值稳定，下降更快更稳
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range (num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features, test_labels))
        print(f'epoch: {epoch} 训练log rmse：{float(train_ls[-1]):f}')
    return train_ls, test_ls

"""K折交叉验证"""
def get_k_fold_data(k, i, X, y):
    """
    获取K折交叉验证 中 第 i 次的数据
    :param k: 分为k折
    :param i: 选择第i个切片作为验证数据集
    """
    assert k > 1
    # // 是整除
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        # 一折的 index
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:  # 验证数据集
            X_valid, y_valid = X_part, y_part
        elif X_train is None:  # 最初的一次，赋值
            X_train, y_train = X_part, y_part
        else:  # 整合训练数据集
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    """
    K折交叉验证
    """
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        # 每次拿第 i 折作验证集
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        # * 是解包
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        # train函数返回的是训练和验证损失数组，数组中每个元素是每一轮计算的损失。这里-1代表只取最后一轮损失
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1,num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}, 训练log rmse{float(train_ls[-1]):f},'
              f'验证log rmse{float(valid_ls[-1]):f}')
    # 求和 取平均
    return train_l_sum / k, valid_l_sum / k

"""模型选择"""
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.01, 0.001, 256

"""提交Kaggle预测"""
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    net.to(device=torch.device('cuda'))
    # 这里没有验证集了
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    d2l.plt.draw()

    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().cpu().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
d2l.plt.show()