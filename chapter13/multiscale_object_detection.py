# %matplotlib inline
import torch
from d2l import torch as d2l

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
# print(h, w)  # 561 728

def display_anchors(fmap_w, fmap_h, s):
    """
    特征图（fmap）上生成锚框（anchors），每个单位（像素）作为锚框的中心
        fmap_w：特征图的宽
        fmap_h：特征图的高
        s：锚框大小，[0,1]
    """
    d2l.set_figsize()
    # 前两个维度上的值不影响输出（批量大小，通道数，高，宽）
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    # 生成以每个像素为中心具有不同形状的锚框
    # （这里传进去的是特征图）（因为坐标用的是比例[0,1]，所以映射回原始图片直接乘宽高即可）
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    # print(anchors.shape)
    # print(anchors)
    # 该锚框的数值范围为 [0,1]，所以要乘以宽高来缩放
    bbox_scale = torch.tensor((w, h, w, h))
    # 映射回原始图像
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)  # 取[0]是因为批量大小为1

# display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
# 特征图的高度和宽度减小一半，然后使用较大的锚框来检测较大的目标
# display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
# 进一步将特征图的高度和宽度减小一半，然后将锚框的尺度增加到0.8
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])


d2l.plt.show()