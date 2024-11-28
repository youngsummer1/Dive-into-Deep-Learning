# %matplotlib inline
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 读取内容和风格图像
d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg')
# d2l.plt.imshow(content_img)

style_img = d2l.Image.open('../img/autumn-oak.jpg')
# d2l.plt.imshow(style_img)

"""预处理和后处理"""
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    """预处理， 图片 --> 可训练的tensor"""
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    """后处理，tensor --> 图片"""
    img = img[0].to(rgb_std.device)
    # 还原到 没做过 normalize 前，且小于0的换成0，大于1的换成1
    # （因为图像打印函数要求每个像素的浮点数值在0～1之间）
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

"""抽取图像特征"""
# 使用基于ImageNet数据集预训练的VGG-19模型来抽取图像特征
# (原论文就用这个)
pretrained_net = torchvision.models.vgg19(weights="DEFAULT")
# 样式层 和 内容层
# 样式层 --> 为匹配局部和全局的风格，从不同层抽取
# 内容层 --> 为了避免合成图像过多保留内容图像的细节，选择VGG较靠近输出的层
style_layers, content_layers = [0, 5, 10, 19, 28], [25]

# 只保留需要用到的VGG的所有层
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])

# 逐层计算，并保留内容层和风格层的输出
def extract_features(X, content_layers, style_layers):
    """逐层计算，并保留内容层和风格层的输出"""
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

# 因为 网络的权重是不变的，所以 特征可以直接抽好，一劳永逸
def get_contents(image_shape, device):
    """对内容图像抽取内容特征"""
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    """对风格图像抽取风格特征"""
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y

"""定义损失函数"""
# 内容损失
def content_loss(Y_hat, Y):
    """内容损失"""
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。

    # 均方误差
    # 两个输入均为extract_features函数计算所得到的内容层的输出
    return torch.square(Y_hat - Y.detach()).mean()

# 风格损失
def gram(X):
    """格拉姆矩阵"""
    # n --> 宽 * 高
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    # 被看作由 num_channels 个长度为 宽 * 高 的向量组合而成的
    # 第 i 行代表了通道 i 上的风格特征
    X = X.reshape((num_channels, n))
    # 格拉姆矩阵， 表达风格层输出的风格
    # 为了让风格损失不受元素个数多少的影响，除以了矩阵中元素的个数
    return torch.matmul(X, X.T) / (num_channels * n)
def style_loss(Y_hat, gram_Y):
    """风格损失"""
    # 均方误差
    # 分别基于合成图像与风格图像的风格层输出
    # 这里假设基于风格图像的格拉姆矩阵gram_Y已经预先计算好了 （避免重复计算）
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

# 全变分损失
def tv_loss(Y_hat):
    """全变分损失，用于降噪"""
    # 比较每个像素和周围像素的绝对值
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

"""损失函数"""
# 格转移的损失函数是内容损失、风格损失和总变化损失的加权和
# （权重的选择，使三个损失在同一个数量级）
content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、风格损失和全变分损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

"""初始化合成图像"""
# 合成的图像是训练期间唯一需要更新的变量
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        # 将合成的图像视为模型参数
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight

def get_inits(X, device, lr, styles_Y):
    """初始化"""
    # 创建了合成图像的模型实例
    gen_img = SynthesizedImage(X.shape).to(device)
    # 将其初始化为图像X （通常为内容图片？）
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    # 风格图像在各个风格层的的gram提前算好，因为后续不会变
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

"""训练模型"""
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        # 前三个损失是print用的，关键是综合的损失 l
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X

# 首先将内容图像和风格图像的高和宽分别调整为300和450像素，用内容图像来初始化合成图像
device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)

d2l.plt.show()