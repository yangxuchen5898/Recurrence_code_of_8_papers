## ResNet——基于深度残差学习的图像识别（2015）

### 摘要

更深的神经网络更难训练。我们提出一种 **残差学习框架** ，用以简化比以往深得多的网络的训练。我们将各层明确地重新表述为：参照图层输出来学习残差函数，而不是学习无参照的函数。我们提供了充分的实证证据表明：这些残差网络更易于优化，而且能够得到很好的精度。

在 ImageNet 这个数据集上，我们使用了 $152$ 层，是 VGG 的 $8$ 倍但是有更低的复杂度。用这些残差网络做了一个整体之后得到了 $3.57 \%$ 的错误率，我们还在 CIFAR-10 上对 $100$ 层与 $1000$ 层网络进行了分析。

特征表示的深度对许多视觉识别任务至关重要。仅凭借我们极深的表示，就在 COCO 目标检测数据集上取得了 $28 \%$ 的相对提升。

### 引言

深度网络能够自然地以端到端的多层方式整合低层/中层/高层特征以及分类器，这些特征的层次能够通过堆叠层数而得到丰富。不同level的层能够得到不同的特征（低层/中层/高层特征）

学习更好地网络就是简单地把更多网络堆在一起吗？然而如果网络特别深就会出现梯度爆炸或者消失。

解决方式是

- 在权重随机初始化的时候不要太大也不要太小
- 在中间加一些归一化层以校验每个层之间的输出和梯度的均值、方差

使用这些后，是可以收敛了，然而在网络变深之后精度变差了。这不是层数变多而导致的过拟合，因为拟合是训练误差低而测试误差高，下图展示的是训练误差和测试误差都高。

为什么加了更多层后精度会变差？如果浅的网络效果可以的话，深的网络效果是不应该变差的，因为多加的层可以使得输入是 $x$ ，输出也是 $x$ ，通过把权重先乘 $\frac{1}{n}$ 再乘 $n$ 来做到。这种做法叫做 identity mapping （恒等映射），但实际上 SGD 做不到。

<img src="C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250911201829637.png" alt="image-20250911201829637" style="zoom: 25%;" />

左图是训练误差，右图是测试误差。使用 $20$ 层和 $56$ 层的没有残差的，可以发现更深的网络错误率反而更高。

于是这篇文章提出一个办法使得能够显式地构造出一个恒等映射（deep residual learning framework），使得深的网络不会变得比浅的网络差。

<img src="C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250911202555694.png" alt="image-20250911202555694" style="zoom:25%;" />

例如原有的层叫 $H(x)$ ，现在在此基础上新加一些层，学习的不再是 $H(x)$ ，而是 $H(x)-x$ ， $F(x)=H(x)-x$ 也就是残差，它的输出是 $F(x)+x$ 。

这样做不会增加任何要学习的参数，也就不会增加模型复杂度，而且这个网络仍然是可以被训练的。

非常深的残差网络非常容易被优化，但是如果不加这个残差连接的话，效果就会很差，且加上残差时越深精度越高。

### 相关工作

**残差表示**

在图像识别中，VLAD 是一种表示方法，它通过相对于字典的残差向量来编码；而 Fisher Vector 可以被表述为 VLAD 的一种概率版本。这两种方法都是强大的浅层表示，用于图像检索和分类。在向量量化任务中，已有研究表明：编码残差向量比编码原始向量更有效。

在低层视觉和计算机图形学中，为了解偏微分方程（PDEs），广泛使用的多重网格（Multigrid）方法将系统重新表述为多尺度上的子问题，其中每个子问题都负责在粗尺度和细尺度之间的残差解。多重网格的一种替代方法是分层基预条件（hierarchical basis preconditioning），它依赖于在两个尺度之间表示残差向量的变量。已有研究表明，这些求解器的收敛速度远快于那些没有考虑残差特性的标准求解器。这些方法表明，一个良好的重新表述或预条件可以简化优化过程。

**捷径连接**

导致捷径连接的实践与理论已经被研究了很长时间。早期在训练多层感知机（MLPs）时的一种做法是，在网络输入和输出之间添加一层线性连接。在 [[44](https://arxiv.org/pdf/1409.4842), [24](https://arxiv.org/pdf/1409.5185)] 中，一些中间层被直接连接到辅助分类器上，用于解决梯度消失/梯度爆炸问题。[[39](https://n.schraudolph.org/pubs/Schraudolph98.pdf), [38](https://nic.schraudolph.org/pubs/facede.pdf), [31](http://yann.lecun.com/exdb/publis/pdf/raiko-aistats-12.pdf), [47](https://arxiv.org/pdf/1301.3476)] 的论文提出了通过捷径连接来居中层响应、梯度和误差传播的方法。在 [[44](https://arxiv.org/pdf/1409.4842)] 中，一个 “Inception” 层由一条捷径分支和若干更深的分支组成。

与我们的工作同时期，“Highway Networks”提出了带有门控函数（gating functions）的捷径连接。这些门控是依赖于数据的，并且带有参数，与我们提出的无参数的恒等捷径（identity shortcuts）不同。当门控捷径“关闭”（接近零）时，高速公路网络中的层就表示非残差函数。相反，在我们的设计中，始终是学习残差函数；我们的恒等捷径从不会关闭，所有信息总是被完整传递，同时还需要学习额外的残差函数。此外，高速公路网络并未在极大深度（如超过 $100$ 层）的情况下展现出精度提升。

### 深度残差学习

#### 残差学习

#### 快捷方式标识映射

#### 网络结构

残差网络如何处理输入和输出的形状是不同的情况

- 方案一

  - 在输入和输出上分别添加一些额外的 $0$ ，使这两个形状能够对应起来，可以做相加
- 方案二
  
  - 选取一个 $1 \times 1$ 的卷积使得输出通道是输入通道的两倍，这样就能把残差链接的输入和输出对比上了
  - 如果我们把输出通道数翻了两倍，，那么输入的高和宽通常都会减半，所以在做 $1 \times 1$ 的卷积时步幅设置为 $2$ ，做出来在高和宽上能够匹配

#### 实现

**一些实验细节**

把短边随机地采样到 $[256, \ 480]$ ，这样我在做后面 $224 \times 224$ 的切割时随机性更大。

图像颜色处理：把每一个像素的均值减掉，还用了一些颜色的增强。

用了批量归一化（BN）。

所有的权重和文献 [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852) 用的是一样的。

批量大小 batch 是 $256$ 。

学习率是 $0.1$ ，每一次除以 $10$ （当错误率比较平稳时就除以 $10$ ）。

迭代 $6 \times 10^5$ 次，算下来训练 $120$ 轮，计算公式如下

>批量大小： $256$
>
>数据集：ImageNet，约 $128$ 万张图像
>
>每个 epoch 的迭代次数 $\approx 1.28M / 256 \approx 5000$
>
>总迭代次数：$600000$ 次
>
>相当于 大约 $120$ 个 epoch

没有用 dropout ，因为没有全连接层。

**一些测试细节**

使用标准的 10-crop 测试

>**10-crop testing 的方法**
>
>“10-crop” 是 ImageNet 等论文里常见的标准测试方式，具体做法是：
>
>1. 从一张测试图像生成 $10$ 个裁剪（crops）：
>   - $5$ 个位置：左上角、右上角、左下角、右下角、中心。
>   - 再加上水平翻转 → 共 $10$ 个裁剪。
>2. 每个裁剪大小固定（例如 $224 \times 224$ ）。
>3. 把 $10$ 个裁剪分别输入模型，得到 $10$ 组预测结果。
>4. 最终预测 = $10$ 组结果的平均值（或投票）。

在 $\{ 224,256,384,480,640 \}$ 这些不同的分辨率上做采样

>**来对比一下 ImageNet 是怎样的**
>
>ImageNet 包含分辨率不一的图像，而我们的系统需要固定的输入尺寸。因此，我们将图像下采样到固定分辨率 $256 \times 256$ 。对于一个矩形图像，我们首先将其缩放，使得较短的一边长度为 $256$ ，然后从缩放后的图像中裁剪出中心的 $256 \times 256$ 区域。除了从每个像素中减去训练集上的平均值外，我们没有对图像进行其他任何预处理。因此，我们的网络是直接在（已中心化的）像素原始 RGB 值上进行训练的。

### 实验

#### ImageNet分类

![image-20250911202753639](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250911202753639.png)

不使用残差的情况下， $34$ 层错误率高于 $18$ 层，使用残差的情况下， $18$ 层错误率高于 $34$ 层。

![image-20250913004832079](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250913004832079.png)

这张表展示了整个 ResNet 不同架构之间的构成，有 $18$ 层， $34$ 层， $50$ 层， $101$ 层， $152$ 层

卷积层、最大池化层、全连接层都是一样的

残差块里输入和输出维度（通道数、高宽）不一致时，shortcut（捷径连接）如何处理。作者比较了三种方案：

(A) Zero-padding shortcuts

- 当输出的维度比输入大时，用 $0$ 填充把输入扩展到相同维度。
- 这样 shortcut 不需要学习参数（parameter-free）。
- 举例：输入有 64 个通道，输出需要 128 个通道，就在输入的右侧补 64 个“全零通道”。
- 优点：计算开销最小；缺点：信息利用率有限。

(B) Projection shortcuts for increasing dimensions, identity otherwise

- 当维度需要增加时，用 投影 shortcut（通常是 $1 \times 1$ 卷积，stride 可>1）来把输入映射到和输出一致的维度。
- 如果维度没变，就用恒等映射（identity shortcut）。
- 这是一个折中方法：保证维度对齐时能学到更合适的变换，同时大多数层仍然是无参数的 identity shortcut。

(C) All shortcuts are projections

- 不论维度是否匹配，所有 shortcut 都用投影层（ $1 \times 1$ 卷积）。
- 这样每条捷径都有可学习参数，灵活性最强。
- 缺点：计算和参数量增加。

##### 构建更深的ResNet

到 $50$ 或者 $50$ 以上的层时会引入一个叫 bottleneck （瓶颈）的设计

<img src="C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250913180437842.png" alt="image-20250913180437842" style="zoom:25%;" />

从图可知，相比左边，右边先把 $256$ 降维为 $64$ ，最后再升为 $256$ 

#### CIFAR-10和分类

#### 基于PASCAL和MS-COCO的目标检测

`model.py` ，使用 $34$ 层

```python
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, Linear, Identity

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, projection=False):
        super(ResidualBlock, self).__init__()

        self.ResidualBlock_model = Sequential(

            Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channels)
        )
        self.relu = ReLU(inplace=True)
        # 捷径：需要下采样或通道变化时用 1x1 投影 + BN；否则恒等
        if projection:
            self.shortcut = Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = Identity()

    def forward(self, x):
        identity = x
        out  = self.ResidualBlock_model(x)
        # 将捷径加到输出上
        out  += self.shortcut(identity)
        out  = self.relu(out )
        return out

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.ResNet_model = Sequential(
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            # CIFAR-10 数据集通常不用 MaxPool2d 因为本身就很小了
            # MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 定义各个阶段的残差块
            self._make_layer(64, 64, 3, stride=1),
            self._make_layer(64, 128, 4, stride=2),
            self._make_layer(128, 256, 6, stride=2),
            self._make_layer(256, 512, 3, stride=2),
        )

        # 分类器
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc = Linear(512, 10)  # ImageNet有1000个类别


    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        proj = (stride != 1) or (in_channels != out_channels)
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, projection=proj))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, projection=False))
        return Sequential(*layers)

    def forward(self, x):
        x = self.ResNet_model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = ResNet34()
    if torch.cuda.is_available():
        model.cuda()
    print(model)
```

`train.py` ，使用 CIFAR-10 数据集

```python
from model import ResNet34
import torchvision
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch import save
writer = SummaryWriter("logs")
model = ResNet34().cuda()
model.train()
train_set = torchvision.datasets.CIFAR10(root = "train_and_test_dataset",
                                         train = True,
                                         download = True,
                                         transform = torchvision.transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle=True)
loss_function = nn.CrossEntropyLoss().cuda()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, weight_decay=5e-4)
# 用变量train_step记录训练的次数
train_step = 0
# 用变量epoch记录训练的次数
epoch = 15
for i in range(epoch):
    print("第", i + 1, "轮")
    running_loss = 0.0
    correct = 0
    total = 0
    for data in train_dataloader:
        optimizer.zero_grad()
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = model(imgs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_step += 1
        _, predicted = outputs.max(1)  # 取最大值所在的类别
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    acc = correct / total
    writer.add_scalar(tag="train_loss", scalar_value=running_loss, global_step=i + 1)
    writer.add_scalar(tag="acc", scalar_value=acc, global_step=i + 1)
    print(running_loss)
    print(acc)
    save(model.state_dict(), "resnet_model_{}.pth".format(i + 1))
writer.close()
```

训练 $15$ 轮，损失和正确率变化如下

```
第 1 轮
1153.2679405808449
0.4629
第 2 轮
703.4594068825245
0.68358
第 3 轮
500.9185376763344
0.77774
第 4 轮
387.99551072716713
0.8276
第 5 轮
314.68024411797523
0.86006
第 6 轮
250.04973653703928
0.8883
第 7 轮
199.6749854646623
0.91018
第 8 轮
160.72337564080954
0.92808
第 9 轮
134.61165308952332
0.93862
第 10 轮
106.84003648534417
0.95112
第 11 轮
94.22974587604403
0.9572
第 12 轮
75.31833736319095
0.96564
第 13 轮
66.73015029774979
0.9703
第 14 轮
62.15779575845227
0.97184
第 15 轮
56.956364979967475
0.97518
```

在tensorboard画图如下

![image-20250915021441935](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250915021441935.png)

![image-20250915021450807](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250915021450807.png)

`test.py`

```python
import torchvision
from model import ResNet34
import torch
device = torch.device("cuda")
test_set = torchvision.datasets.CIFAR10(root = "train_and_test_dataset",
                                        train = False,
                                        download = True,
                                        transform = torchvision.transforms.ToTensor())
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle=False)
model = ResNet34().to(device)
model.eval()
state = torch.load("resnet_model_15.pth", map_location=device)
msg = model.load_state_dict(state, strict=True)
print("load_state_dict report:", msg)
with torch.no_grad():
    correct = 0
    total = 0
    for data in test_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        _, predicted = outputs.max(1)  # 取最大值所在的类别
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    acc = correct / total
    print(acc)
```

输出显示正确率 $0.8212$

