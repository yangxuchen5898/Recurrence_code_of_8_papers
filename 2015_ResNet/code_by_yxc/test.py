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