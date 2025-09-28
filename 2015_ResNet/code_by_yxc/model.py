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
