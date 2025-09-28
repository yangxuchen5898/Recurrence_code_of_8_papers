import os
import glob
import numpy as np
import imageio.v2 as imageio
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model import Unet
def load_and_reflect_pad(p):
    img = imageio.imread(p) # (512,512) 或 (512,512,C)
    if img.ndim == 3: # 若是RGB，先取灰度/某一通道
        img = img[..., 0]
    t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0) # (1,1,512,512)
    t = F.pad(t, (30,30,30,30), mode='reflect') # (1,1,572,572)
    return t
class ISBIDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __getitem__(self, idx):
        return self.images[idx].float(), self.labels[idx].float()
    def __len__(self):
        return len(self.images)
def center_crop(t, size_hw):
    _, _, h, w = t.shape
    th, tw = size_hw
    i = (h - th) // 2
    j = (w - tw) // 2
    return t[:, :, i:i+th, j:j+tw]
if __name__ == '__main__':
    writer = SummaryWriter("logs")
    device = torch.device("cuda")
    train_image_paths = sorted(glob.glob(os.path.join(r".\dataset\ISBI-2012-Challenge\train\imgs", "*.png")))
    train_label_paths = sorted(glob.glob(os.path.join(r".\dataset\ISBI-2012-Challenge\train\labels", "*.png")))
    train_images = torch.cat([load_and_reflect_pad(p) for p in train_image_paths], dim=0)
    train_images = train_images / 255.0
    train_labels = torch.cat([load_and_reflect_pad(p) for p in train_label_paths], dim=0)
    train_labels = train_labels / 255.0
    train_dataset = ISBIDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=1)
    unet = Unet().to(device)
    unet.train()
    loss_function = nn.BCEWithLogitsLoss().to(device)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(unet.parameters(), lr = learning_rate, momentum=0.99)
    epoch = 10
    for i in range(epoch):
        print("第", i + 1, "轮")
        train_loss = 0.0
        for data in train_loader:
            optimizer.zero_grad()
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = unet(imgs)  # N×C×Hout×Wout
            targets_cropped = center_crop(targets, outputs.shape[-2:])
            loss = loss_function(outputs, targets_cropped)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=i + 1)
        print(train_loss)
        torch.save(unet.state_dict(), "unet_model_{}.pth".format(i + 1))
    writer.close()