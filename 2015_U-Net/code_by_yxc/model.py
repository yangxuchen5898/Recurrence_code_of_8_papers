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
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        return self.Unet_model(x)

if __name__ == '__main__':
    unet = Unet()
    if torch.cuda.is_available():
        unet.cuda()
    print(unet)