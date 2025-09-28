## U-Net——生物医学图像的卷积网络（2015）

### 摘要

#### 描述

一种依赖于运用 **数据增强** 的网络和训练策略

#### 网络作用

使我们更有效地运用 **带有注解的** 样例

#### 架构及其作用

包含 **收缩路径** 来 **捕获上下文** 和 **对称扩展路径** 来 **实现精准定位**

#### 网络能力

- 在 **神经元结构图像分割** 中，能够通过 **非常少的图像** 进行 **端到端** 的训练，且胜过此前最佳的方法（滑动窗口卷积网络）
- 在 **透射光显微镜图像** 的训练中大幅度领先
- 网络很快

### 引言

- 由于可用的训练集的大小和考虑的网络大小，卷积神经网络的成功受限。其[突破](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)在于使用 $8$ 层和百万参数的大型网络的 **监督** 训练。
- 卷积神经网络的典型应用是分类任务，图像的输出是 **单个** 类标签。然而在很多视觉任务中尤其是生物图像处理中，需要的输出应当 **包含位置信息** ，也就是说分类标签应该被 **分配到每一个像素** 。于是在[滑动窗口](https://papers.nips.cc/paper_files/paper/2012/file/459a4ddcb586f24efd9395aa7662bc7c-Paper.pdf)中训练网络，通过提供围绕该像素的局部 **区域/补丁** 作为输入来预测每个像素的类标签。但也存在缺点： **速度太慢** 以及 **在位置准确性和使用上下文之间有权衡** 。
  - 大补丁需要更多 **最大池化层数** ，这会降低位置准确性
  - 小补丁只能看到一点点上下文
  - 解决方法[1](https://openaccess.thecvf.com/content_cvpr_2015/papers/Hariharan_Hypercolumns_for_Object_2015_CVPR_paper.pdf)与[2](https://openaccess.thecvf.com/content_iccv_2013/papers/Seyedhosseini_Image_Segmentation_with_2013_ICCV_paper.pdf)是一种考虑多层特征的分类器输出，能够兼顾这两点
- 我们建立在一个[架构](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)之上，修改和扩展它使得我们能进行 **更少的训练** 和产生 **更精准的分割** 。这个架构是在传统 **收缩型卷积网络** 的基础上，通过增加连续的 **上采样层** 进行补充，使得网络输出可以逐步恢复空间分辨率（即把卷积操作后的一丢丢恢复回去），从而实现像素级的语义分割。在这个过程中，原本用于压缩空间信息的池化操作，在输出阶段被上采样操作所取代。我们所作出的修改如下：
  - 上采样部分仍有很多特征通道，使得网络能够将上下文信息传播到高分辨率的层
  - 结果就是，扩展路径和收缩路径基本对称，产生U形的结构
  - 这个网络没有全连接层，只使用每个卷积层的有效部分，也就是说分割图中 **只包含那些在输入图像中能获得完整上下文信息的像素** 
    - **通俗解释：** 想象你正在用放大镜看地图上的某个区域，你的 **眼睛范围有限** ，看清楚一个地方（比如小镇）你需要周围的信息（山脉、河流、公路）来判断它的归属。如果你看的区域太靠边，就可能 **看不全周围的环境** ，就没法准确判断它属于哪个地区。U-Net的卷积网络在每次卷积时都有一个“感受野”（receptive field），意思是： **网络看某个像素的时候，是通过它周围的一块区域来做判断的** 。
    - **U-Net 中这句话的意思是：** 在U-Net的结构中，由于网络的下采样（压缩）和上采样（恢复）操作，网络在靠近图像边缘的区域， **看不全像素周围的信息** ，所以这些像素就不能被准确分割。因此，论文的处理方法是： **只在图像中间那一块输出分割结果** ，这部分的每一个像素都有“完整的上下文”支持它的判断；而边缘那些缺乏上下文的像素，则不包括在分割结果里。
    - **图像层面直观理解：** U-Net结构中，输入图像是例如 $572 \times 572$ 的大小，但最终输出的分割图是 $388 \times 388$ —— **少了一圈边缘像素** 。这是因为：
      - 原图中靠边的像素，在经过卷积、池化后感受野不足，没法给出可靠的语义判断；
      - 所以 U-Net 只保留了 **中间部分** 的分割输出，这部分每个像素都拥有充分的上下文。

  - 该策略允许通过 **重叠平铺策略** 对任意大的图像进行无缝分割。
    - 为了预测图像边界区域中的像素，通过镜像输入图像来外推缺失的上下文。
    - 在下面单开了一个标题详细讲（由ChatGPT生成）

- 我们可用的训练数据非常少，我们通过把弹性变形应用到可用的训练图像中，以此来使用过度的数据增强。
  - 在下面单开了一个标题详细讲（由ChatGPT生成）

- 另一个在很多细胞分割任务中的挑战是分离同一类的触摸对象。
  - 为此，我们建议使用加权损失，其中，在触摸单元之间分离的背景标签在损失函数中获得较大权重。
  - 在下面单开了一个标题详细讲（由ChatGPT生成）


### 网络架构

![image-20250726063555987](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250726063555987.png)

- 左边是收缩路径，是典型的卷积神经网络的结构
- 右边是扩展路径
- 包含两个 $3 \times 3$ 卷积的应用（未使用padded的卷积），每一个后面都跟随有一个ReLU激活函数和一个 $2 \times 2$ 的最大池化操作用于 **下采样** ，步长为 $2$
- 在每一个下采样步骤我们将特征通道的数量乘 $2$
- 在扩展路径的每一步包含了一个跟随着一个 $3 \times 3$ 卷积的特征图的上采样，能够将特征通道的数量减半，这个上采样是一个串联伴随着相应裁剪的从收缩路径而来的特征图和两个每个都跟随有一个ReLU激活函数的 $3 \times 3$ 卷积
- 由于在每一个卷积中边界像素的丢失，所以裁剪是有必要的
- 在最后一层，一个 $1 \times 1$ 卷积将每64个分量特征向量映射到所需的类数
- 整体上这个网络有 $23$ 个卷积层
- 为了让输出的分割图像（如图2）能够是无缝补丁，选择合适的输入补丁的大小是很重要的，例如所有的 $2 \times 2$ 的最大池化操作都作用在 $x$ 和 $y$ 尺寸均为偶数的层上

#### 代码复现（by myself）

`model.py`

```python
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, ConvTranspose2d
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.Unet_model = Sequential(
            Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0),
            Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0),
            Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0),
            Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        return self.Unet_model(x)
if __name__ == '__main__':
    unet = Unet()
    if torch.cuda.is_available():
        unet.cuda()
    print(unet)
```

### 训练

输入的图像和它们相应的分割图像被用于训练网络，我们用[Caffe框架](https://arxiv.org/pdf/1408.5093)中提供的随机梯度下降算法实现来训练网络。由于卷积是没有填充的，输出图像比输入图像更小，少掉固定宽度的边框区域。为了最小化开销且最大化使用GPU的显存，我们喜欢更大的图块而不是大的批量，因此将批量缩减为单张图像。相应地我们使用高动量系数（ $0.99$ ），这样在当前的优化步骤中，更新会由之前看到的大量训练样本共同决定。

能量函数（下称“损失函数”）在最终的特征图上进行逐像素的 `softmax` 计算，并结合交叉熵损失函数来计算。 `softmax` 被定义为
$$
p_k(\mathbf{x}) = \frac{\exp(a_k(\mathbf{x}))}{\sum_{k'=1}^K \exp(a_{k'}(\mathbf{x}))}
$$
其中， $a_k(\mathbf{x})$ 表示在像素位置 $\mathbf{x} \in \Omega(\Omega \subset \mathbb{Z}^2)$ 处，第 $k$ 个特征通道的激活值。 $K$ 是类别总数， $p_k(\mathbf{x})$ 是最大函数的近似值。也就是说，有最大激活 $a_k(\mathbf{x})$ 的 $k$ 对应有 $p_k(\mathbf{x}) \approx 1$ ，其他 $k$ 对应有 $p_k(\mathbf{x}) \approx 0$ 。交叉熵然后会在每一个位置惩罚 $p_{\ell(\mathbf{x})}(\mathbf{x})$ 到 $1$ 的偏差，使用公式
$$
E = \sum_{\mathbf{x} \in \Omega} w(\mathbf{x}) \log \big( p_{\ell(\mathbf{x})}(\mathbf{x}) \big)
$$
其中， $\ell$ 是每一个像素的正确标签， $\ell \in \mathbb{Z}$ ， $w$ 是一个权重图，我们引入它来在训练中给一些像素更大权重， $w \in \mathbb{R}$ 。

我们为每个真实标注的分割结果预先计算权重图，以补偿训练数据集中某一类别像素出现频率的差异，并强制网络学习我们在相邻细胞之间引入的细小分隔边界（见图 3c 和 3d）。

分界线使用形态学运算来计算，权重图计算公式如下
$$
w(\mathbf{x}) = w_c(\mathbf{x}) + w_0 \cdot \exp\left( -\frac{(d_1(\mathbf{x}) + d_2(\mathbf{x}))^2}{2\sigma^2} \right)
$$
其中， $w_c$ 是权重图来平衡类别出现的频率， $w_c \in \mathbb{R}$ ， $d_1$ 表示到最近细胞边界的距离， $d_2$ 表示到第二近细胞边界的距离。在我们的实验中，我们设置 $w_0 = 10$ 且 $\sigma \approx 5$ 个像素。

在有很多卷积层和不同路径穿过网络的深度网络中，优秀的权重的初始化是十分重要的。否则，部分网络可能会过度激活，同时其他部分从来不贡献。理想情况下初始化权重应该是合适的，这样的话网络中的每一个特征图能够有近似单位方差。对于一个有我们的结构（交替卷积和ReLU层）的网络，这能够通过通过从高斯分布中提取初始权重来得到，标准差为 $\sqrt{\frac{2}{N}}$ ，其中， $N$ 表示一个神经元的传入节点数。例如，对于上一层的 $64$ 特征通道和 $3 \times 3$ 卷积层， $N=9 \times 64=576$ 

#### 代码复现（by myself，还没有使用数据增强）

`train.py`

```python
import os
import glob
import numpy as np
import imageio.v2 as imageio
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model import Unet
def load_and_reflect_pad(p):
    img = imageio.imread(p) # (512,512) 或 (512,512,C)
    if img.ndim == 3: # 若是RGB，先取灰度/某一通道
        img = img[..., 0]
    t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0) # (1,1,512,512)
    t = F.pad(t, (30,30,30,30), mode='reflect') # (1,1,572,572)
    return t
class ISBIDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __getitem__(self, idx):
        return self.images[idx].float(), self.labels[idx].float()
    def __len__(self):
        return len(self.images)
def center_crop(t, size_hw):
    _, _, h, w = t.shape
    th, tw = size_hw
    i = (h - th) // 2
    j = (w - tw) // 2
    return t[:, :, i:i+th, j:j+tw]
if __name__ == '__main__':
    writer = SummaryWriter("logs")
    device = torch.device("cuda")
    train_image_paths = sorted(glob.glob(os.path.join(r".\dataset\ISBI-2012-Challenge\train\imgs", "*.png")))
    train_label_paths = sorted(glob.glob(os.path.join(r".\dataset\ISBI-2012-Challenge\train\labels", "*.png")))
    train_images = torch.cat([load_and_reflect_pad(p) for p in train_image_paths], dim=0)
    train_images = train_images / 255.0
    train_labels = torch.cat([load_and_reflect_pad(p) for p in train_label_paths], dim=0)
    train_labels = train_labels / 255.0
    train_dataset = ISBIDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=1)
    unet = Unet().to(device)
    loss_function = nn.BCEWithLogitsLoss().to(device)
    learning_rate = 0.01
    optimizer = torch.optim.Adam(unet.parameters(), lr = learning_rate)
    epoch = 10
    for i in range(epoch):
        print("第", i + 1, "轮")
        train_loss = 0.0
        for data in train_loader:
            optimizer.zero_grad()
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = unet(imgs)  # N×C×Hout×Wout
            targets_cropped = center_crop(targets, outputs.shape[-2:])
            loss = loss_function(outputs, targets_cropped)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=i + 1)
        print(train_loss)
        torch.save(unet.state_dict(), "my_first_model_{}.pth".format(i + 1))
    writer.close()
```

训练 $10$ 轮，损失变化如下

```
第 1 轮
120828241127.15598
第 2 轮
1242783831.7542129
第 3 轮
2802959866.5859375
第 4 轮
3055940127.734375
第 5 轮
25263312.42233008
第 6 轮
16.231099367141724
第 7 轮
16.135781854391098
第 8 轮
16.157793790102005
第 9 轮
16.155325651168823
第 10 轮
16.156940698623657
```

在tensorboard画图如下

![image-20250811202636825](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250811202636825.png)

#### 数据增强

当训练样例很少能够得到时，为了教导网络所需的不变性和鲁棒性属性，数据增强是必要的。对于显微镜图像，我们主要需要平移和旋转不变性，以及对形变和灰度值变化的鲁棒性。尤其是对训练样本进行随机弹性形变，似乎是用极少的标注图像训练分割网络的关键概念。

我们通过在粗略的 $3 \times 3$ 网格上使用随机位移向量来生成平滑形变。位移值从标准差为 $10$ 个像素的高斯分布中采样，然后使用双三次插值计算每个像素的位移。收缩路径末端的Dropout层还能进一步执行隐式数据增强。

### 实验

我们展示了U-Net在三种不同分割任务中的应用。

第一个任务是对电子显微镜记录中的神经元结构进行分割。数据集的一个示例以及我们得到的分割结果显示在图2中。完整结果作为 **补充材料** 提供。数据集来自[ISBI 2012开始的EM分割挑战赛](https://www.kaggle.com/datasets/hamzamohiuddin/isbi-2012-challenge)，并且至今仍对新的贡献开放。训练数据由 $30$ 张图像（ $512 \times 512$ 像素）组成，这些图像来自果蝇（Drosophila）一龄幼虫腹神经索（VNC）的连续切片透射电子显微镜。每张图像都配有对应的、完整标注的真实分割图——细胞用白色表示，细胞膜用黑色表示。测试集是公开可获取的，但其分割标注是保密的。评测方法是将预测得到的细胞膜概率图发送给主办方。评测过程会在 $10$ 个不同阈值下对概率图进行二值化，并计算“形变误差”（warping error）、“Rand 误差”（Rand error）和“像素误差”（pixel error）。

U-Net（对输入数据的 $7$ 种旋转版本取平均）在没有进行任何额外的预处理或后处理的情况下，取得了 $0.0003529$ 的形变误差（新的最佳成绩，见表 $1$）以及 $0.0382$ 的Rand误差。

这一结果显著优于Ciresan等人提出的滑动窗口卷积网络方法，他们的最佳提交结果的形变误差为 $0.000420$，Rand误差为 $0.0504$。在Rand误差方面，唯一表现更好的算法是在该数据集上使用了与数据集高度相关的特定后处理方法，这些方法是应用在Ciresan等人的概率图上的。

我们还将U-Net应用于光学显微镜图像中的细胞分割任务。该分割任务是ISBI细胞追踪挑战赛2014和2015的一部分。第一个数据集“PhC-U373”包含在聚丙烯酰胺基底上培养的多形性胶质母细胞瘤-星形细胞瘤U373细胞，通过相差显微镜拍摄（见图4a、b以及补充材料）。该数据集包含 $35$ 张部分标注的训练图像。在该任务中，我们获得了平均IOU（“交并比”）92%的成绩，这明显优于第二名算法的83%（见表2）。第二个数据集“DIC-HeLa”是通过微分干涉相衬（DIC）显微镜在平面玻璃上拍摄的HeLa细胞（见图3、图4c、d以及补充材料）。该数据集包含20张部分标注的训练图像。在该任务中，我们获得了平均IOU77.5%的成绩，这明显优于第二名算法的46%。

#### 代码复现（by myself）

`test.py`

```python
import glob
import os
from torch.utils.data import Dataset, DataLoader
from model import Unet
import torch
from train import load_and_reflect_pad, center_crop, ISBIDataset
device = torch.device("cuda")
unet = Unet().to(device)
test_image_paths = sorted(glob.glob(os.path.join(r".\dataset\ISBI-2012-Challenge\test\imgs", "*.png")))
test_label_paths = sorted(glob.glob(os.path.join(r".\dataset\ISBI-2012-Challenge\test\labels", "*.png")))
test_images = torch.cat([load_and_reflect_pad(p) for p in test_image_paths], dim=0)
test_images = test_images / 255.0
test_labels = torch.cat([load_and_reflect_pad(p) for p in test_label_paths], dim=0)
test_labels = test_labels / 255.0
test_dataset = ISBIDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1)
state = torch.load("my_first_model_10.pth", map_location=device)
msg = unet.load_state_dict(state, strict=True)
print("load_state_dict report:", msg)
unet.to(device).eval()
correct_total = 0
pixel_total = 0
with torch.no_grad():
    for imgs, targets in test_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = unet(imgs)
        targets_cropped = center_crop(targets, outputs.shape[-2:])
        if outputs.shape[1] == 1:
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
            targets_idx = (targets_cropped > 0.5).long().squeeze(1)
        else:
            preds = outputs.argmax(1)
            targets_idx = (targets_cropped > 0.5).long().squeeze(1)
        correct_total += (preds == targets_idx).sum().item()
        pixel_total += preds.numel()
acc = correct_total / pixel_total
print(acc)
```

输出显示正确率72%

```
load_state_dict report: <All keys matched successfully>
0.7215602525950331
```

### 结论

U-Net架构在各种截然不同的生物医学分割应用中都取得了非常出色的表现。得益于采用弹性形变的数据增强，它只需要极少的标注图像，并且在NVidia Titan GPU（6GB）上训练时间仅需 $10$ 小时，训练效率非常合理。我们提供了基于Caffe的完整实现以及训练好的网络。我们确信，U-Net架构能够轻松应用于更多的任务中。

###  重叠平铺策略（Overlap-Tile Strategy）

![image-20250726045657097](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250726045657097.png)

这张图展示的是 **U-Net** 论文中提出的一个非常重要的技术策略，叫做： **Overlap-Tile Strategy（重叠平铺策略）**

#### 一、图中元素解释

- **左边图像：** 原始灰度图，表示的是输入的一张大图像（如神经元图像）。
- **蓝色框：** 表示一个 **输入补丁（tile）** ，即神经网络一次处理的一个图像块。
- **黄色框：** 表示该tile中 **最终输出的分割区域**，即segmentation map。
- **右边图像：** 网络预测的分割结果图。可以看到黄色区域变成了二值分割图的对应区域。

#### 二、问题：为什么不能直接对整张图做分割？

现实中，我们经常遇到 **特别大的图像** （比如医学图像、遥感图像），但神经网络（如U-Net）的输入大小是有限的，比如只能处理 $572 \times 572$ 。

因此，必须把大图 **切成一小块一小块地送进网络** ，每块处理完以后再拼起来。但是：

##### 如果直接切割：

- 那么在每个tile边缘的像素，由于 **周围上下文信息不足** ，预测效果会变差。
- 并且由于上下文裁掉了，导致边缘会出现拼接缝隙（block artifact）。

#### 三、解决方案：Overlap-Tile Strategy（重叠块策略）

这张图展示了一个策略： **每个 tile（蓝色框）预测的只是中间的区域（黄色框），而输入的时候“故意多给一圈”上下文边界。**

##### 步骤如下：

1. **从整张图中取出一块蓝色区域** 作为输入（比预测区域更大）。
2. **网络只输出中间的黄色区域** 的分割结果（因为只有这些像素的“感受野”是完整的）。
3. **这样可以保证预测结果质量较高，没有边缘模糊问题**。
4. 为了覆盖整张图， **所有蓝色 tile 是有重叠的** ，就像地砖错位铺设一样，称为overlap。
5. **最终所有黄色区域的结果拼在一起，形成完整的输出图像**。

#### 四、边界处理：边缘数据不足怎么办？

图中还提到一句话：“Missing input data is extrapolated by mirroring.”

即：如果蓝色输入框靠近图像边缘，外部的信息不存在怎么办？ **镜像填充（mirroring）**

例如：如果蓝色框超出了原图边界，就把图像边缘 **镜像翻转一部分 ** 填进去，凑够感受野。

#### 总结

**Overlap-tile strategy** 是一种将大图像分割成重叠小块处理的技巧， 每块只预测中心区域，边缘通过镜像填充，最终将所有中心预测块拼接还原为整图分割结果。 这样做的目的，是保证每个预测像素都拥有 **完整的上下文信息** ，提高预测精度并避免边缘失真。

### 弹性形变（elastic deformation）

#### 这么做的动因

U-Net是为医学图像分割设计的，但医学图像的 **标注成本非常高** ，一个医生标一张图可能要几小时，所以数据量通常 **很小** ，而深度学习又需要 **大量数据** 才能学得好。于是他们采用了一个策略来“扩充数据集”： **“数据增强”（Data Augmentation）** 。人为制造出“新的训练图”，本质上是对已有图像进行各种“变化处理”。

**弹性形变（elastic deformation）** 就是他们采用的一种非常有效的增强方式。

#### 什么是弹性形变

你可以把它想象成这样一个比喻：把图像像橡皮布一样，轻轻拉伸、扭曲、抖动一下。虽然图像的内容整体没变，但局部细节的位置发生了细微扰动，就好像是新的样本了。

具体做法常用的是：

- 在图像上施加一个 **随机位移场（displacement field）** ；
- 然后用插值的方式生成一张新的图像；
- 这可以模拟真实世界中的“组织轻微变形”“拍摄角度差异”等变化。

论文中说这是 **最重要的数据增强方式** ，比什么旋转、平移、缩放、翻转都更有效，特别适合医学图像。

#### 用Python生成几张“弹性形变”前后的图示例（by ChatGPT）

```python
from skimage import data
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates
# 读取并调整大小
image = data.camera()
image = resize(image, (128, 128), anti_aliasing=True)
def elastic_deformation(image, alpha, sigma):
    """对图像施加弹性形变"""
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    distorted_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return distorted_image
# 生成弹性形变图像
deformed1 = elastic_deformation(image, alpha=20, sigma=3)
deformed2 = elastic_deformation(image, alpha=40, sigma=5)
# 绘制图像
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = ['原图像', '弹性形变 α=20, σ=3', '弹性形变 α=40, σ=5']
images = [image, deformed1, deformed2]
for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

![image-20250726053606628](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250726053606628.png)

#### 总结

由于医学图像数据稀缺，U-Net作者通过“橡皮筋式地扭曲图像”的方式来人为制造更多训练样本，这种弹性形变能保持图像结构大致不变，同时让网络学到更多“变形不变性”，是U-Net成功的重要技巧之一。

### 加权损失函数（weighted loss）

![image-20250726055553658](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250726055553658.png)

多个细胞紧贴在一起时很难区分边界，U-Net通过“加权损失函数”强化模型对边界像素的学习，让它学会“把贴在一起的细胞分开”。

#### 图像各部分说明：

| 图    | 内容                            | 含义                                                         |
| ----- | ------------------------------- | ------------------------------------------------------------ |
| **a** | 原始图像                        | 细胞显微镜图像，灰度图                                       |
| **b** | 原图 + 标注                     | 不同颜色表示不同的细胞实例（Ground Truth）                   |
| **c** | 分割掩码                        | 白色是细胞（前景），黑色是背景，可以看到边界细细一条线也是黑色（→ 关键） |
| **d** | 像素级权重图（loss weight map） | 用于 **加权损失函数** ，红色区域表示权重高（在细胞边界处），蓝色区域权重低 |

#### 背后的问题：为什么要加权？

在细胞分割中，尤其是HeLa细胞这类任务里，常见的问题是：

- **多个细胞紧贴在一起** ；
- 网络很容易把它们分成一个整体（合在一起），而不是一个一个单独的细胞。

如果我们只是用普通的损失函数，比如交叉熵，那么：

- 网络不会特别关心细胞之间的“边界”；
- 它更容易把多个“挤在一起”的细胞看成一个。

#### U-Net的解决方案：加权损失函数

对图像中的每个像素分配一个权重 `w(x)` ，在计算损失时， **边界像素的损失被乘以更大的权重** ，让网络“特别关注边界”。

##### 在图 d 中：

- **红色** （权重大）：细胞之间的分割线，表示网络需要重点学习这里；
- **蓝色** （权重小）：普通背景或细胞内部区域，网络不需要花太多精力。

#### 数学上的权重定义（简化表达）：

在U-Net论文中，作者为每个像素设计了一个权重函数（本文中的公式2）：
$$
w(x) = w_c(x) + w_0 \cdot \exp\left( -\frac{(d_1(x) + d_2(x))^2}{2\sigma^2} \right)
$$
其中：

- $w_c(x)$ ：用于处理类不平衡（比如前景比背景少）
- $d_1(x), d_2(x)$ ：是当前像素到最近的两个细胞边界的距离
- 这意味着： **越靠近两个细胞之间的“缝隙”，权重越大**

#### 用Python复现这类加权map，模拟不同loss权重的效果（by ChatGPT）

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.draw import disk
from skimage.color import label2rgb
rng = np.random.default_rng(2025)
# ------------------------------------------------------------------
# 1) 生成 8 个互相接触的圆形“细胞”实例标签
# ------------------------------------------------------------------
H, W = 256, 256
instance = np.zeros((H, W), int)
centers = np.array([[80, 80], [80, 140], [140, 50],
                    [140, 110], [140, 170], [200, 100],
                    [60, 200], [200, 60]])
radii = np.full(len(centers), 40)
for idx, (cy, cx) in enumerate(centers, 1):
    rr, cc = disk((cy, cx), radii[idx-1], shape=instance.shape)
    instance[rr, cc] = idx
# ------------------------------------------------------------------
# 2) 生成“原始灰度显微图”
#    - 每个细胞一个基准灰度 (0.55~0.95)
#    - 加少量空间相关噪声 + 高斯模糊
# ------------------------------------------------------------------
raw = np.zeros((H, W), float)
for lbl in range(1, instance.max()+1):
    base = rng.uniform(0.55, 0.95)          # 每个细胞一个灰度
    mask = instance == lbl
    raw[mask] = base
# 背景稍微暗一点
raw[instance == 0] = rng.uniform(0.1, 0.25)
# 加轻微随机纹理噪声
raw += rng.normal(0, 0.03, size=raw.shape)
raw = gaussian_filter(raw, sigma=2)         # 模糊逼近显微成像
raw = np.clip(raw, 0, 1)
# ------------------------------------------------------------------
# 3) Ground‑truth 叠加彩色图
# ------------------------------------------------------------------
overlay = label2rgb(instance, image=raw, bg_label=0, alpha=0.45)
# ------------------------------------------------------------------
# 4) 二值掩码 & 细边界
# ------------------------------------------------------------------
binary = (instance > 0).astype(int)
boundary = (np.roll(instance, 1, 0) != instance) | (np.roll(instance, 1, 1) != instance)
binary_thin = binary.copy()
binary_thin[boundary] = 0  # 把边界像素设成 0
# ------------------------------------------------------------------
# 5) U‑Net 加权图  —  不屏蔽内部 ⇒ 环形高权
# ------------------------------------------------------------------
labels = np.unique(instance)[1:]
dist_stack = np.stack([distance_transform_edt(instance != l) for l in labels])
d1, d2 = np.sort(dist_stack, axis=0)[:2]
w0, sigma = 10, 7.0
weight_map = 1 + w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))
# ------------------------------------------------------------------
# 6) 绘制 4 幅结果
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
axes[0].imshow(raw, cmap='gray')
axes[0].set_title('(a) 原始灰度显微图\n（每个细胞灰度略不同）', fontsize=9)
axes[0].axis('off')
axes[1].imshow(overlay)
axes[1].set_title('(b) 叠加标注', fontsize=9)
axes[1].axis('off')
axes[2].imshow(binary_thin, cmap='gray')
axes[2].set_title('(c) 二值掩码', fontsize=9)
axes[2].axis('off')
im = axes[3].imshow(weight_map, cmap='jet', vmin=1, vmax=1+w0)
axes[3].set_title('(d) 像素级权重图', fontsize=9)
axes[3].axis('off')
plt.colorbar(im, ax=axes[3], fraction=0.045, pad=0.04)
plt.tight_layout()
plt.show()
```

![](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250726063432493.png)

#### 总结

U-Net的核心技巧之一：当两个细胞靠得很近时，它会 **人为加大这两者之间“空隙”的训练权重** ，让模型特别注意分清它们的边界，从而实现更好的分割效果。

### 插值算法

#### 最近邻插值（Nearest Neighbor Interpolation）

将每个目标像素找到距离它最近的原图像素点，然后将该像素的值直接赋值给目标像素。

优点：实现简单，计算速度快

缺点：插值结果缺乏连续性，可能会产生锯齿状的边缘，对于图像质量的影响较大。

因此，当处理精度要求较高的图像时，通常会采用更加精细的插值算法，如双线性插值、双三次插值等

例如，要把 $3 \times 3$ 的图像扩成 $5 \times 5$ 的，先把 $5 \times 5$ 的和 $3 \times 3$ 的缩放为一样的大小

<img src="C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250811060211647.png" alt="image-20250811060211647" style="zoom:50%;" />

然后将两图重合，看 $5 \times 5$ 的图像的每一个格子落在 $3 \times 3$ 的图像中的哪一个格子最多，就取几

<img src="C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250811060446998.png" alt="image-20250811060446998" style="zoom: 33%;" />

#### 双线性插值



### 名词解释（由ChatGPT生成）

#### 收缩路径

##### 定义

收缩路径是 U-Net 网络的左半部分，负责逐步提取输入图像中的“上下文信息（context）”。

##### 具体结构

- 典型模块：
   每一级由 **两个 3×3 卷积（valid）+ ReLU**，再加一个 **2×2 的 MaxPooling（stride=2）**

- 每进行一次下采样，**图像尺寸减半，通道数翻倍**：

  ```
  输入 (572x572x1)
      ↓ Conv-ReLU-Conv-ReLU
  输出 (568x568x64)
      ↓ MaxPool (2x2)
  输出 (284x284x64) → 下一层继续
  ```

##### 功能目的

- 最大程度压缩图像空间信息（resolution），
- 同时捕获更大的**感受野（receptive field）**，
- 提取高级语义特征：例如，整个细胞、膜、核的上下文含义

------

#### 对称扩展路径

##### 定义

扩张路径是 U-Net 网络的右半部分，负责将收缩路径压缩的信息逐步“还原”到原图大小，以实现**像素级分割**。

##### 具体结构

- 每一级由：
  1. 一个 **2×2 的 Up-Convolution（转置卷积）**：图像大小扩大2倍，通道减半
  2. 然后将**对应层的收缩路径特征图**通过**拼接（Concatenation）**并联入通道中（这就是“跳跃连接”）
  3. 接着做两个 **3×3 卷积 + ReLU** 处理融合后的特征

例如：

```
上采样：(28x28x512) → (56x56x256)
拼接跳跃特征：(56x56x256) + (56x56x256) → (56x56x512)
卷积处理 → 得到更精细、语义明确的特征
```

##### 功能目的

- 将从收缩路径中提取的高级特征“融合回”原图尺寸，帮助定位目标边缘与细节
- 实现精准的像素级分类：既有**语义上下文（深层）**，又保留**空间位置（浅层）**

----

#### 端到端训练

##### 定义

“End-to-end training” 指的是整个神经网络从输入到输出的所有参数**一体化训练**，**不需要人为地拆分成多个阶段或模块**来分别训练。

##### 可以这样理解

###### 非 End-to-End 的传统流程（早期图像处理）：

1. 手动做**边缘检测**
2. 接着用**人工特征提取器**（如SIFT、HOG）
3. 最后用一个**分类器**（如SVM）来判断结果

这三步分别训练、分开优化，无法形成一个统一优化目标。

###### End-to-End 的现代深度学习（如 U-Net）：

- 输入：原始图像
- 输出：每个像素的类别（细胞、背景等）
- 中间过程：由网络自行学习（卷积、特征提取、融合、上采样…）
- 所有参数**统一用一个损失函数反向传播优化**
- 用户不需要对中间过程作出任何人为干预

模型从“头到尾”都在优化一个目标 —— 最终的像素分类结果。

###### U-Net 中的 End-to-End 应用

在这篇论文中，“U-Net can be trained end-to-end” 具体指：

- 从原始图像输入 → 到最终分割掩膜输出
- **整个网络结构（收缩路径+扩张路径）** 是一个连续计算图
- 通过一套 loss 函数（加权交叉熵）+ softmax 输出 → 用反向传播训练整个模型

##### 优点总结

| 优点             | 解释                                       |
| ---------------- | ------------------------------------------ |
| 无需人工特征工程 | 特征提取由网络自动学习完成                 |
| 统一优化目标     | 训练过程更一致、更有效                     |
| 推理速度更快     | 不需分阶段执行多个模型                     |
| 更易部署         | 可直接将整个网络打包部署在临床、工业系统中 |

