import glob
import os
from torch.utils.data import Dataset, DataLoader
from model import Unet
import torch
from train import load_and_reflect_pad, center_crop, ISBIDataset
device = torch.device("cuda")
unet = Unet().to(device)
unet.eval()
test_image_paths = sorted(glob.glob(os.path.join(r".\dataset\ISBI-2012-Challenge\test\imgs", "*.png")))
test_label_paths = sorted(glob.glob(os.path.join(r".\dataset\ISBI-2012-Challenge\test\labels", "*.png")))
test_images = torch.cat([load_and_reflect_pad(p) for p in test_image_paths], dim=0)
test_images = test_images / 255.0
test_labels = torch.cat([load_and_reflect_pad(p) for p in test_label_paths], dim=0)
test_labels = test_labels / 255.0
test_dataset = ISBIDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1)
state = torch.load("unet_model_10.pth", map_location=device)
msg = unet.load_state_dict(state, strict=True)
print("load_state_dict report:", msg)
unet.to(device).eval()
correct_total = 0
pixel_total = 0
with torch.no_grad():
    for imgs, targets in test_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = unet(imgs)
        targets_cropped = center_crop(targets, outputs.shape[-2:])
        if outputs.shape[1] == 1:
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
            targets_idx = (targets_cropped > 0.5).long().squeeze(1)
        else:
            preds = outputs.argmax(1)
            targets_idx = (targets_cropped > 0.5).long().squeeze(1)
        correct_total += (preds == targets_idx).sum().item()
        pixel_total += preds.numel()
acc = correct_total / pixel_total
print(acc)
