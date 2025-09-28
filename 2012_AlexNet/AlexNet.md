## AlexNet——基于深度卷积神经网络的图像网络分类（2012）

### 摘要

我们训练了一个大型、深层的卷积神经网络，用来将 ImageNet LSVRC-2010 比赛中的 120 万张高分辨率图像分类到 1000 个不同的类别中。在测试数据上，我们分别取得了 top-1 错误率 37.5% 和 top-5 错误率 17.0%，这比当时之前的最佳结果要好得多。该神经网络拥有 6000 万个参数和 65 万个神经元，由五个卷积层（其中一些后接最大池化层）以及三个全连接层组成，最后是一个 1000 类的 softmax 层。为了加快训练速度，我们使用了非饱和型神经元以及一种非常高效的 GPU 卷积运算实现。为了减少全连接层的过拟合，我们采用了一种新近开发的正则化方法——“dropout”，事实证明其效果非常好。我们还将该模型的一个变体参加了 ILSVRC-2012 比赛，并取得了 top-5 测试错误率 15.3% 的获胜成绩，而亚军的成绩是 26.2%。

### 引言

当前的目标识别方法在很大程度上依赖于机器学习方法。为了提升性能，我们可以收集更大的数据集、学习更强大的模型，并采用更好的防止过拟合的技术。直到最近，带标注的图像数据集还相对较小——数量级在几万张图像左右（例如 NORB、Caltech-101/256、CIFAR-10/100）。对于简单的识别任务，这样规模的数据集已经可以很好地解决问题，尤其是在结合保持标签不变的图像变换进行数据增强时。例如，在 MNIST 手写数字识别任务上，当前最佳错误率（小于 0.3%）已经接近人类表现。然而，在真实环境中的物体具有很大的变化性，要学会识别它们就必须使用更大的训练集。实际上，小规模图像数据集的局限性早已被广泛认识到，但直到最近才有可能收集到包含数百万张图像的标注数据集。新的大规模数据集包括 LabelMe（包含数十万张完全分割的图像）和 ImageNet（包含超过 1500 万张带标签的高分辨率图像，覆盖 22000 多个类别）。

为了从数百万张图像中学习成千上万种物体，我们需要一个具有大规模学习能力的模型。然而，目标识别任务的巨大复杂性意味着，即便是像 ImageNet 这样庞大的数据集，也无法完全刻画这一问题，因此我们的模型还应具备大量的先验知识，以弥补缺失的数据。卷积神经网络（CNN）就是这样一类模型。它们的容量可以通过调整深度和宽度来控制，并且它们对图像的性质（即统计特性的平稳性以及像素依赖的局部性）做出了强有力且大体正确的假设。因此，与具有相似规模层数的标准前馈神经网络相比，CNN 的连接数和参数量要少得多，因此更易于训练，同时其理论上的最佳性能通常只会稍微差一些。

尽管 CNN 具有许多吸引人的优点，并且其局部结构相对高效，但在大规模高分辨率图像上应用时，它们的计算成本依然高得令人望而却步。幸运的是，当前的 GPU 与高度优化的二维卷积实现相结合，已经足够强大，可以支持训练规模相当大的 CNN。同时，像 ImageNet 这样的最新数据集也提供了足够多的标注样本，使得在不发生严重过拟合的情况下训练此类模型成为可能。

本文的具体贡献如下：我们在 ILSVRC-2010 和 ILSVRC-2012 比赛中使用的 ImageNet 子集上，训练了迄今为止最大规模的卷积神经网络之一，并在这些数据集上取得了迄今为止远超以往的最佳结果。我们编写了一个高度优化的 GPU 二维卷积实现，以及训练卷积神经网络所需的其他所有操作，并将其公开发布。我们的网络包含若干新的、非同寻常的特性，这些特性提升了性能并缩短了训练时间（详见第 3 节）。由于网络规模庞大，即便有 120 万带标签的训练样本，过拟合依然是一个显著的问题，因此我们使用了多种有效的防止过拟合的技术（详见第 4 节）。我们的最终网络包含 5 个卷积层和 3 个全连接层，这样的深度似乎非常重要：我们发现，即便去掉任意一个卷积层（每个卷积层的参数量占模型总参数量不到 1%），都会导致性能下降。

最终，网络的规模主要受限于当前 GPU 可用的显存容量，以及我们所能容忍的训练时间。我们的网络在两块 GTX 580 3GB GPU 上训练需要 5 到 6 天。我们所有的实验都表明，只要等待更快的 GPU 和更大的数据集出现，我们的结果就能得到进一步提升。

### 数据集

ImageNet 是一个包含超过 1500 万张带标签的高分辨率图像的数据集，这些图像大约覆盖 22000 个类别。图像是从网络上收集的，并通过亚马逊 Mechanical Turk 众包工具由人工进行标注。自 2010 年起，作为 Pascal 视觉目标挑战赛的一部分，每年都会举办一个名为 ImageNet 大规模视觉识别挑战赛（ILSVRC）的比赛。ILSVRC 使用 ImageNet 的一个子集，每个类别大约包含 1000 张图像，总计约 120 万张训练图像、5 万张验证图像和 15 万张测试图像。

ILSVRC-2010 是唯一一个测试集标签可用的 ILSVRC 版本，因此我们的大部分实验都是在这一版本上进行的。由于我们也参加了 ILSVRC-2012 比赛，所以在第 6 节中我们也报告了该版本数据集上的结果，但该版本的测试集标签是不可用的。在 ImageNet 上，通常会报告两种错误率：top-1 和 top-5。其中，top-5 错误率指的是测试图像中，那些真实标签不在模型预测的最可能的五个标签之列的比例。

ImageNet 包含分辨率不一的图像，而我们的系统需要固定的输入尺寸。因此，我们将图像下采样到固定分辨率 256 $\times$ 256。对于一个矩形图像，我们首先将其缩放，使得较短的一边长度为 256，然后从缩放后的图像中裁剪出中心的 256 $\times$ 256 区域。除了从每个像素中减去训练集上的平均值外，我们没有对图像进行其他任何预处理。因此，我们的网络是直接在（已中心化的）像素原始 RGB 值上进行训练的。

### 结构

我们网络的架构如图 2 所示。它包含八个需要学习的层——五个卷积层和三个全连接层。下面，我们将描述网络结构中一些新颖或不常见的特性。第 3.1 节到第 3.4 节按照我们认为的重要性排序，最重要的放在最前面。

#### ReLU激活函数

原先的激活函数使用的是
$$
f(x)=\tanh(x)
$$
或
$$
f(x)=(1+e^{-x})^{-1}
$$
本文使用ReLU激活函数，防止过拟合且更快

#### 多GPU训练

#### 局部响应归一化（Local Response Normalization）

$$
b_{x,y}^i = \frac{a_{x,y}^i}  {\left( k + \alpha \sum_{j=\max(0,\, i-n/2)}^{\min(N-1,\, i+n/2)} \left( a_{x,y}^j \right)^2 \right)^{\beta}}
$$

```python
torchvision.datasets.ImageNet(root: Union[str, Path], split: str = 'train', **kwargs: Any)
```

`model.py`

```python
from torch import nn
from torch.nn import Conv2d, LocalResponseNorm, MaxPool2d, Flatten, Dropout, Linear, ReLU
class ALEXNET(nn.Module):
    def __init__(self):
        super(ALEXNET, self).__init__()
        self.AlexNet_model = nn.Sequential(
            Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            ReLU(inplace=True),
            LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(96, 256, kernel_size=5, padding=2),
            ReLU(inplace=True),
            LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(256, 384, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(384, 384, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Flatten(),
            Dropout(0.5),
            Linear(256 * 6 * 6, 4096),
            ReLU(inplace=True),
            Dropout(0.5),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Linear(4096, 1000)
        )
    def forward(self, x):
        x = self.AlexNet_model(x)
        return x
if __name__ == '__main__':
    alexnet = ALEXNET().cuda()
    print(alexnet)
```

`train.py`

```python
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from model import ALEXNET
from torch import nn
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(
    r"E:\迅雷下载\ImageNet\data\ImageNet2012\train_data",
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)
if __name__ == '__main__':
    writer = SummaryWriter("logs")
    device = torch.device("cuda")
    alexnet = ALEXNET().to(device)
    alexnet.train()
    loss_function = nn.CrossEntropyLoss().to(device)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(alexnet.parameters(), lr=learning_rate, momentum = 0.9)
    epoch = 15
    for i in range(epoch):
        print("第", i + 1, "轮")
        train_loss = 0.0
        for data in train_loader:
            optimizer.zero_grad()
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = alexnet(imgs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=i + 1)
        print(train_loss)
        torch.save(alexnet.state_dict(), "alexnet_model_{}.pth".format(i + 1))
    writer.close()
```

```
第 1 轮
4963.3957715034485
第 2 轮
4214.583386421204
第 3 轮
3499.375861644745
第 4 轮
3027.845828771591
第 5 轮
2693.9509139060974
第 6 轮
2410.9413628578186
第 7 轮
2201.711982488632
第 8 轮
2028.4005899429321
第 9 轮
1881.2213522195816
第 10 轮
1755.4850187301636
第 11 轮
1655.4002009630203
第 12 轮
1566.174175977707
第 13 轮
1483.489560008049
第 14 轮
1421.7108897566795
第 15 轮
1359.6820585727692
```

![image-20250816072814956](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250816072814956.png)

`test.py`

```python
import torch
from torchvision import transforms, datasets
from train import normalize
def test_data_loader(valdir):
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )
    return val_loader
```

测试集没有标签测不了，所以没测

