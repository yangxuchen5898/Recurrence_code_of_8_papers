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