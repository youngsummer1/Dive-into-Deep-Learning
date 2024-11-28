# %matplotlib inline
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

"""类别预测层"""
# 对每个像素做预测
def cls_predictor(num_inputs, num_anchors, num_classes):
    """
    对每个像素 预测锚框的类别 的层
    :param num_inputs: 输入通道数
    :param num_anchors: 每个像素生成的锚框数
    :param num_classes: 类别数
    """
    # 输出通道数 = 每个像素生成的锚框数 * （类别数 + 1 ）   (加1是背景类)
    # 这里针对的是每个像素！！！对每个像素做预测
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)  # 保持高宽不变

"""边框预测层"""
def bbox_predictor(num_inputs, num_anchors):
    """
    对每个像素 预测跟真实边界框的offset （偏移值是四个值）
    :param num_inputs:
    :param num_anchors: 每个像素生成的锚框数
    """
    # 每个锚框的offset有四个值，所以乘 4
    return nn.Conv2d(num_inputs, num_anchors * 4,
                     kernel_size=3, padding=1)  # 保持高宽不变

"""连结多尺度的预测"""

def forward(x, block):
    """将 x 输入块，返回其输出"""
    return block(x)
# 输出通道数 = 5 * (10 + 1) = 55
# 即对输入的20 * 20 = 400像素，每个像素都要做 55 个预测
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
# 输出通道数 = 3 * (10 + 1) = 33
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
# print(Y1.shape, Y2.shape)

# 为了将这两个预测输出链接起来以提高计算效率，我们将把这些张量转换为更一致的格式
def flatten_pred(pred):
    """转换成 （批量大小，高 * 宽 * 通道数）"""
    # 通道维放到最后，是为了在拉直时，把预测同一个像素的锚框放在一起
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

# 合在一起，变成一个大的tensor，以免还要根据不同分辨率分别算
# (方便后续处理)
def concat_preds(preds):
    """在维度 1 上的连结  （变长）"""
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# print(concat_preds([Y1, Y2]).shape)  # torch.Size([2, 25300])

"""高和宽减半块"""

def down_sample_blk(in_channels, out_channels):
    blk = []
    # 卷积 + batchnorm + relu ，重复两次
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))  # 不改变高宽
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    # 高宽为 2 的最大池化，默认stride也为 2
    # 高宽减半
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
# 会更改输入通道的数量，并将输入特征图的高度和宽度减半
# print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)


"""基本网络快"""
def base_net():
    """基本网络块用于从输入图像中 抽取特征"""
    blk = []
    num_filters = [3, 16, 32, 64]
    # 构造 3 个 高宽减半块 (高宽减 8 倍)
    # 通道数 从 3 --> 16 --> 32 --> 64
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

# print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)

"""完整的模型"""
def get_blk(i):
    """完整的单发多框检测模型"""
    # 由五个模块组成
    # 每个块生成的特征图既用于生成锚框，又用于预测这些锚框的类别和偏移量
    if i == 0:  # 第一个是基本网络块
        blk = base_net()
    elif i == 1:  # 第二个到第四个是高和宽减半块
        blk = down_sample_blk(64, 128)
    elif i == 4:  # 最后一个模块使用 全局最大池 将高度和宽度都降到1
        blk = nn.AdaptiveMaxPool2d((1,1))
    else: # 第二个到第四个是高和宽减半块
        blk = down_sample_blk(128, 128)  # 维持通道数在 128
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    """
    为每个块定义前向传播
    :param X: 输入
    :param blk: block,本身是个network
    :param size: 锚框大小
    :param ratio: 锚框宽高比
    :return: （特征图，锚框，每个锚框类别预测，锚框到真实边缘框偏移的预测）
    """
    # 特征图 Y
    Y = blk(X)
    # 生成锚框
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    # 下面两个预测为了简单，参数在构造的时候已经传进去了（所以这里不用传） （152行左右）
    # 在forward的时候不关心锚框长什么样子，backward 算loss的时候才管
    # 每个锚框类别预测
    cls_preds = cls_predictor(Y)
    # 锚框到真实边缘框偏移的预测
    bbox_preds = bbox_predictor(Y)
    # （特征图，锚框，每个锚框类别预测，锚框到真实边缘框偏移的预测）
    return (Y, anchors, cls_preds, bbox_preds)

# 超参数
# 第一个值：锚框大小的比例       第二个值：根号（当前锚框大小 * 下一个锚框大小）
# （不懂？？）
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
# 宽高比， 常用 1 2 0.5
# 乘 5 ，是有 5 个stage
ratios = [[1, 2, 0.5]] * 5
# 每个像素生成的锚框数 = 2 + 3 - 1 = 4
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

# 定义完整的模型
class TinySSD(nn.Module):
    """定义完整的模型"""
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        # 每个 block 的 输入通道数
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            # （感觉像一个宏定义）
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))  # 这里直接在构造的时候就传进去了
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        # 每个 block 的都要存，最后合并起来
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        # 连结，并在一起
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        # reshape 成 3d， 最后维度为 对每个类别的预测值 （softmax方便点）
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        # （锚框，对类别的预测，对偏移量的预测）
        return anchors, cls_preds, bbox_preds

# 创建一个模型实例，使用它对一个 256 * 256 像素的小批量图像X执行前向传播
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)
# （32 * 32 + 16 * 16 + 8 * 8 + 4 * 4 + 1 * 1） * 4 = 5444
# print('output anchors:', anchors.shape)  # torch.Size([1, 5444, 4])
# print('output class preds:', cls_preds.shape)  # torch.Size([32, 5444, 2])
# 5444 * 4 （每个锚框做四个预测）（偏移要用四个值表示）
# print('output bbox preds:', bbox_preds.shape)  # torch.Size([32, 21776])


"""训练模型"""
# 读取数据集和初始化
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# 定义损失函数和评价函数
# 分类 --> 用 交叉熵损失函数
cls_loss = nn.CrossEntropyLoss(reduction='none')
# 回归问题，使用L1 （不用L2，因为可能预测的很远，防止loss过大）
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    """
    损失
    :param cls_preds: 类别预测
    :param cls_labels: 真实标注
    :param bbox_preds: 偏移预测
    :param bbox_labels: 真实标注
    :param bbox_masks: 令负类锚框和填充锚框为 0，不参与损失的计算
    """
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),  # （批量大小 * 锚框维，类别数）
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,  #
                     bbox_labels * bbox_masks).mean(dim=1)
    # 两个损失相加（可以加权）
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    """准确率评估"""
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    """准确率评估"""
    # 使用平均绝对误差来评价边界框的预测结果
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# 训练模型
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    # target 是真实边缘框
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量  （锚框 和 真实边缘框 对应）
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')

"""预测目标"""
# 读取并调整测试图像的大小，然后将其转成卷积层需要的四维格式
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    # 将 cls_preds 做 softmax
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    # 可以根据锚框及其预测偏移量得到预测边界框。然后，通过非极大值抑制来移除相似的预测边界框
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    # 删除 nms 去掉的框
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)

# 筛选所有置信度不低于0.9的边界框，做为最终输出
def display(img, output, threshold):
    """筛选所有置信度不低于 threshold 的边界框，做为最终输出"""
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)

d2l.plt.show()