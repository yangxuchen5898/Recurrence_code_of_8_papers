import torch
from torch import nn
from torch.nn import Sequential, ConvTranspose2d, BatchNorm2d, ReLU, Tanh, Conv2d, LeakyReLU, Sigmoid

class Generator(nn.Module):
    def __init__(self, z_dim=64):
        super(Generator, self).__init__()
        self.GAN_Generator = Sequential(
            ConvTranspose2d(z_dim, 256, 4, 2, 1, bias=False),
            BatchNorm2d(256),
            ReLU(True),
            ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            BatchNorm2d(128),
            ReLU(True),
            ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            BatchNorm2d(64),
            ReLU(True),
            ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            BatchNorm2d(32),
            ReLU(True),
            ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            Tanh()
        )
    def forward(self, x):
        return self.GAN_Generator(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.GAN_Discriminator = Sequential(
            Conv2d(3, 64, 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            Conv2d(64, 128, 4, 2, 1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(0.2, inplace=True),
            Conv2d(128, 256, 4, 2, 1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.2, inplace=True),
            Conv2d(256, 1,kernel_size=4),
            Sigmoid()
        )
    def forward(self, x):
        return self.GAN_Discriminator(x)

if __name__ == '__main__':
    generator = Generator()
    discriminator = Discriminator()
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
    print(generator)
    print(discriminator)