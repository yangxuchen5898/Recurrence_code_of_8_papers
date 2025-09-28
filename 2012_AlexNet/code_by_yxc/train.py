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