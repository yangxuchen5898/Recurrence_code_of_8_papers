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
# 测试集没有标签测不了