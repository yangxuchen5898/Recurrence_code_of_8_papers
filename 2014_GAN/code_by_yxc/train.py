import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model import Generator, Discriminator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision.models import inception_v3, Inception_V3_Weights

# 权重初始化
# 初始化神经网络中的权重，特别是卷积层和批量归一化层的权重
def weights_init(m):
    # 检查 m 是否是卷积层或反卷积层
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        # 使卷积层的权重保持正态分布
        init.xavier_normal_(m.weight)
    # 检查 m 是否是批量归一化层
    elif type(m) == nn.BatchNorm2d:
        # 将批量归一化层的权重设为均值为 1 ，标准差为 0.02 的正态分布
        init.normal_(m.weight, 1.0, 0.02)
        # 将批量归一化层的偏置设为 0
        init.constant_(m.bias, 0)

# 计算某个图片文件夹的 Inception Score（IS），用来评估生成的图像“质量+多样性”
# # IS是用来评价生成模型的
# def inception_score(img_folder, batch_size=32, splits=10):
#     # 加载预训练的 Inception v3 模型
#     weights = Inception_V3_Weights.IMAGENET1K_V1
#     model = inception_v3(weights=weights, aux_logits=True)
#     model = model.to(device)
#     model.eval()
#     preprocess = weights.transforms()
#     # 图片预处理
#     transform = transforms.Compose([
#         # Inception v3 输入大小为 299x299
#         transforms.Resize(299), transforms.CenterCrop(299),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化到 [-1, 1]，这个参数适合训练GAN
#         # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # 这个参数适合ImageNet模型的训练，到时候替换对比一下
#     ])
#
#     # 加载图片数据
#     image_list = []
#     for file_name in os.listdir(img_folder):
#         if file_name.endswith(('.png', '.jpg', '.jpeg')):  # 检查文件类型
#             img_path = os.path.join(img_folder, file_name)
#             image = Image.open(img_path).convert('RGB')  # 打开图片
#             image = preprocess(image)
#             image_list.append(image)
#     images = torch.stack(image_list)  # 将所有图片堆叠成一个 tensor (N, 3, H, W)
#
#     # 创建数据加载器
#     dataset = TensorDataset(images)
#     # 把你生成的图片文件夹里的图片（不是训练数据集）打包成一个数据集，然后送到 Inception v3 模型里做前向推理
#     # 评估时，不需要随机打乱图片顺序，顺序无关紧要
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#
#     # 获取预测概率
#     preds = []
#     with torch.no_grad():
#         for batch in dataloader:
#             batch = batch.to(device)
#             pred = model(batch)
#             if hasattr(pred, "logits"):  # 新版: InceptionOutputs
#                 logits = pred.logits
#             elif isinstance(pred, (tuple, list)):  # 中间版: (logits, aux_logits)
#                 logits = pred[0]
#             else:  # 老版: Tensor
#                 logits = pred
#             # Numpy 不支持 GPU Tensor , 必须先转到CPU再numpy()
#             preds = torch.cat(logits, dim=0).cpu().numpy()  # 转为 numpy 数组
#
#     # 计算 Inception Score
#     N = preds.shape[0]
#     split_indices = np.array_split(np.arange(N), splits)
#     split_scores = []
#     for idxs in split_indices:
#         part = preds[idxs]  # (n_k, 1000)
#         py = np.mean(part, axis=0, keepdims=True)  # (1, 1000)
#         kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
#         kl = np.sum(kl, axis=1)  # (n_k,)
#         split_scores.append(np.exp(np.mean(kl)))
#
#     return float(np.mean(split_scores)), float(np.std(split_scores))

class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform):
        self.files = [os.path.join(folder, f)
                      for f in sorted(os.listdir(folder))
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {folder} to compute Inception Score.")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 用 with 保证文件句柄及时关闭，避免句柄累计占用资源
        with Image.open(self.files[idx]) as im:
            im = im.convert('RGB')
            im = self.transform(im)
        return im

def inception_score(img_folder, batch_size=32, splits=10):
    # 1) 模型与预处理（新API，无弃用警告）
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, aux_logits=True).to(device).eval()
    preprocess = weights.transforms()  # 包含 Resize(299)+CenterCrop+ToTensor+ImageNet Normalize

    # 2) 流式数据集与 DataLoader（关键：不要先 stack 所有图片）
    dataset = ImageFolderDataset(img_folder, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Windows 建议 0

    # 3) 前向：取主 logits → softmax
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            out = model(batch)
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            preds.append(F.softmax(logits, dim=1).cpu())
    preds = torch.cat(preds, dim=0).numpy()  # (N, 1000)

    # 4) 计算 IS
    N = preds.shape[0]
    split_indices = np.array_split(np.arange(N), splits)
    split_scores = []
    for idxs in split_indices:
        part = preds[idxs]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = np.sum(kl, axis=1)
        split_scores.append(np.exp(np.mean(kl)))
    return float(np.mean(split_scores)), float(np.std(split_scores))

# 定义了训练时的超参数，使用命令行参数解析（argparse）获取它们的值
# def get_hyperparameters():
#     parser = argparse.ArgumentParser(description="Training Hyperparameters")
#     parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
#     parser.add_argument('--learning_rate_G', type=float, default=0.0002, help='Learning rate')
#     parser.add_argument('--learning_rate_D', type=float, default=0.0001, help='Learning rate')
#     parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
#     args = parser.parse_args()
#     args.loss_history = []
#     args.success_rate_history = []
#     args.train_rate_history = []
#     return args
# hp = get_hyperparameters()

def main():
    # 不使用超参数的写法
    num_epochs = 50
    learning_rate_G = 2e-4
    learning_rate_D = 2e-4
    batch_size = 64
    z_dim = 64
    k = 1 # 先训练k轮的D，再训练一轮的G，尝试发现k=2不好，所以试试1
    num_is_samples = 1000
    outdir = './generated_images'
    os.makedirs(outdir, exist_ok=True)
    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    # 训练数据
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    # 定义模型
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


    generator = Generator(z_dim=z_dim).to(device) # z_dim是生成器的输入噪声向量的维度
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    generator.train()
    discriminator.train()

    # 定义优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D, betas=(0.5, 0.999))

    # 生成一个固定的随机噪声张量用于可视化，用来在训练过程中监控生成器的学习进度
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    # 二元交叉熵损失函数
    criterion = nn.BCEWithLogitsLoss()

    # 分别保存 G 和判别器 D 的损失值，通常用来在训练过程中绘制损失曲线
    G_losses = []
    D_losses = []
    iters = 0

    epochs = num_epochs
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            real_images, _ = data
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            # 训练判别器D
            for x in range(k):
                discriminator.zero_grad()
                # 真样本
                real_output = discriminator(real_images)
                real_labels = torch.ones_like(real_output, device=device)
                d_loss_real = criterion(real_output, real_labels)
                # D_x = real_output.mean().item()
                D_x = torch.sigmoid(real_output).mean().item()
                # 假样本
                noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
                fake_images = generator(noise)
                fake_output = discriminator(fake_images.detach())
                fake_labels = torch.zeros_like(fake_output, device=device)
                d_loss_fake = criterion(fake_output, fake_labels)
                # D_G_z1 = fake_output.mean().item()
                D_G_z1 = torch.sigmoid(fake_output).mean().item()
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()

            # 训练生成器G
            generator.zero_grad()
            fake_output_for_g = discriminator(fake_images)
            g_labels = torch.ones_like(fake_output_for_g, device=device)
            g_loss = criterion(fake_output_for_g, g_labels)
            # D_G_z2 = fake_output_for_g.mean().item()
            D_G_z2 = torch.sigmoid(fake_output_for_g).mean().item()
            g_loss.backward()
            optimizer_G.step()

            # 统计 & 可视化

            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            grid = vutils.make_grid(fake, padding=2, normalize=True)
            vutils.save_image(grid, os.path.join(outdir, f'grid_iter_{iters:06d}.png'))


            print(f'[{epoch + 1}/{num_epochs}][{i}/{len(trainloader)}] '
                    f'Loss_D: {d_loss.item():.4f}  Loss_G: {g_loss.item():.4f}  '
                    f'D(x): {D_x:.4f}  D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            iters += 1

            # 每个 epoch 末尾也存一张
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        grid = vutils.make_grid(fake, padding=2, normalize=True)
        vutils.save_image(grid, os.path.join(outdir, f'grid_epoch_{epoch + 1:03d}.png'))

    # ----------------------------
    # 训练结束后：生成用于 IS 的图片并保存
    # ----------------------------
    generator.eval()
    print(f"Generating {num_is_samples} images to folder: {outdir}")
    saved = 0
    bs = max(32, batch_size)
    with torch.no_grad():
        while saved < num_is_samples:
            cur = min(bs, num_is_samples - saved)
            z = torch.randn(cur, z_dim, 1, 1, device=device)
            imgs = generator(z).cpu()  # [-1,1]
            # 反归一化到 [0,1] 再保存
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
            for j in range(imgs.size(0)):
                vutils.save_image(imgs[j], os.path.join(outdir, f'gen_{saved + j:05d}.png'))
            saved += cur
    generator.train()

    # ----------------------------
    # 计算 Inception Score
    # ----------------------------
    mean_is, std_is = inception_score(outdir, batch_size=32, splits=10)
    print(f"Inception Score: {mean_is:.4f} ± {std_is:.4f}")


if __name__ == "__main__":
    main()
