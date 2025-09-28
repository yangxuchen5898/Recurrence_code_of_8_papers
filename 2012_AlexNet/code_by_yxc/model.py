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