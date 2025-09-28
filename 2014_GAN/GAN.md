## GAN——生成对抗网络（2014）

### 摘要

同时训练两个模型：一个捕获数据分布的生成模型 $G$ ，和一个估计样本是否来自训练数据而非 $G$ 的判别模型 $D$ 。生成模型 $G$ 的训练过程是最大化 $D$ 犯错误的概率。

在任意函数空间中，存在唯一的解，使得生成模型 $G$ 恢复训练数据分布，并且判别模型 $D$ 在所有地方等于 $\frac{1}{2}$ 。

即：最终当生成模型 $G$ 完全学到了真实数据的分布时，判别模型 $D$ 将无法区分真实数据和生成数据，输出的值会在所有样本上都趋近于 $0.5$ （即随机猜测）。这意味着生成模型 $G$ 已经成功地生成了与真实数据几乎完全相同的样本，而判别模型 $D$ 不能再判断哪个是真实的哪个是生成的，因为它们几乎无法区分了。

在 $G$ 和 $D$ 由多层感知机定义的情况下，整个系统可以通过 **反向传播** 进行训练。在训练或生成样本时， **无需使用任何马尔科夫链或展开的近似推理网络** 。

### 引入

深度生成模型的影响较小，因为生成模型在最大似然估计和相关策略中需要近似许多不可解的概率计算，并且由于在生成背景下难以利用分段线性单元的优势，因此面临更多挑战。我们提出了一种新的生成模型估计方法，能够绕过这些困难。

在提出的对抗网络（adversarial nets）框架中，生成模型与一个对手进行博弈：判别模型学习判断一个样本是来自模型分布还是来自数据分布。生成模型可以被类比为一队伪造者，试图制造假币并不被察觉，而判别模型则类似于警察，试图识别伪造的货币。在这个博弈中，竞争促使两队不断改进他们的方法，直到伪造品无法与真品区分开来。

这个框架可以为多种模型和优化算法提供特定的训练算法。在本文中，我们探索了一个特例，即 **生成模型通过将随机噪声传递通过多层感知机来生成样本，而判别模型也是一个多层感知机** 。我们将这个特例称为对抗网络（adversarial nets）。在这种情况下，我们只需使用高度成功的 **反向传播和dropout算法** 来训练这两个模型，并且 **仅通过前向传播从生成模型中采样** 。无需使用任何近似推理或马尔科夫链。

### 相关工作

$$
\lim_{\sigma \to 0} \nabla_x E_{\epsilon \sim \mathcal{N}(0, \sigma^2 I)} f(x + \epsilon) = \nabla_x f(x)
$$

随机变量 $\epsilon$ 服从 $(0,\sigma^2)$ 的高斯分布， $\epsilon$ 是噪声

$E_{\epsilon \sim \mathcal{N}(0, \sigma^2 I)}$ 表示对 $\epsilon$ 进行期望操作，期望操作指的是在给定的概率分布下，计算 $\epsilon$ 的平均值

$f(x + \epsilon)$ 表示对 $x$ 加上噪声 $\epsilon$ 后，函数 $f$ 的值

### 对抗网络

这个框架最简单的应用是当生成器和辨别其都是 MLP 时，生成器需要去学一个 在数据 $x$ 上的 $p_g$ 分布。

把 $x$ 比喻为显示器上看到的一张 800 万像素的图片，每一个像素是一个随机变量， $x$ 就是 800 万维的一个多维随机变量，每一个像素的值都是 $p_g$ 分布来控制的

生成模型如何输出 $x$ ？我们定义一个先验（prior）在噪音变量 $p_z(z)$ 上

$z$ 是一个 800 万维的向量，每一个元素服从均值为 $0$ ，方差为 $1$ 的高斯分布的高斯噪音

生成模型就是把 $z$ 映射成 $x$ ，生成模型 $G(z;\theta_g)$ 是一个 MLP ，$\theta_g$ 是他的可学习参数

想要生成图片，可以通过反汇编代码，由此知道这个图片是怎么生成的；也可以学一个映射（MLP），构建一个 800 万维的向量，然后MLP强行把 $z$ 映射成那些我想要的 $x$ 。

辨别模型 $D(X;\theta_d)$ 也是一个 MLP ，$\theta_d$ 是他的可学习参数，作用是把输入一个 800 万像素的图片，输出一个 $0 \sim 1$ 的标量（概率）用于表示是生成模型生成的图片到真实存在的图片的概率

训练 $G$ 的同时也训练 $D$
$$
\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log(1 - D(G(z)))].
$$
公式解析：

$z$ 是随机噪音， $G(z)$ 是生成的图片，如果辨别器完美，则 $D(G(z))=0$ ，此时 $\log(1 - D(G(z)))=0$ 。如果辨别器做的不是很好，则 $0<D(G(z))<1$ ， $\log(1 - D(G(z)))<0$ 。训练 $G$ 的目标就是让辨别器尽可能犯错，从而让 $\log(1 - D(G(z)))$ 尽可能小

$x$ 是真实存在的图片，如果辨别器完美，则 $D(x)=1$ ， $\log D(x)=0$ 。

所以，在辨别器完美的情况下， $\log D(x)$ 与 $\log(1 - D(G(z)))$ 都应该等于 $0$ ，如果辨别器做的不是很好，则 $\log D(x)$ 与 $\log(1 - D(G(z)))$ 都是负数值。

所以 $D$ 要让函数值尽可能大（最大为 $0$）， $G$ 为了让 $D$ 犯错，是让函数值尽可能小。

#### 算法1

![image-20250902095554894](C:\Users\yangxuchen\AppData\Roaming\Typora\typora-user-images\image-20250902095554894.png)

做 $k$ 步，每一步中都是先采样 $m$ 个噪音样本 $\{ z^{(1)} , \cdots , z^{(m)} \}$ ，再采样 $m$ 个真实样本 $\{ x^{(1)} , \cdots , x^{(m)} \}$ ，组成 $2m$ 大小的批量，放进辨别器
$$
\frac{1}{m} \sum_{i=1}^{m} \left[ \log D(x^{(i)}) + \log \left( 1 - D(G(z^{(i)})) \right) \right]
$$
求梯度

然后放进生成器
$$
\frac{1}{m} \sum_{i=1}^{m} \log \left( 1 - D(G(z^{(i)})) \right)
$$
求梯度

可以看到每次迭代是先更新辨别器，再更新生成器

$k$ 是一个超参数，不能太大也不能太小

如果太大，则会将辨别器更新得太好，导致此时下面 $G$ 无论怎么更新， $D(G(z^{(i)})$ 改变都不大。如果将辨别器训练到完美，则 $\log \left( 1 - D(G(z^{(i)})) \right)=0$ ，对等于 $0$ 的东西求导毫无意义

如果太小，则不会促进生成器的更新

更新 $G$ 时最大化 $D(G(z))$ ，防止 $D(G(z))=0$ 使得 $\log(1-D(G(z)))=0$ 而造成梯度消失

### 代码复现

使用 DCGAN 论文的基于 CNN 的方式

`model.py`

```python
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
```

`train.py`

 ```python 
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
 ```

前10行的输出数据

```
[1/50][0/782] Loss_D: 1.4032  Loss_G: 0.6854  D(x): 0.6174  D(G(z)): 0.5983 / 0.5039
[1/50][1/782] Loss_D: 1.2735  Loss_G: 0.6889  D(x): 0.6101  D(G(z)): 0.5392 / 0.5021
[1/50][2/782] Loss_D: 1.2155  Loss_G: 0.6888  D(x): 0.6212  D(G(z)): 0.5212 / 0.5022
[1/50][3/782] Loss_D: 1.1863  Loss_G: 0.6886  D(x): 0.6352  D(G(z)): 0.5176 / 0.5023
[1/50][4/782] Loss_D: 1.1830  Loss_G: 0.6889  D(x): 0.6399  D(G(z)): 0.5192 / 0.5021
[1/50][5/782] Loss_D: 1.1684  Loss_G: 0.6896  D(x): 0.6569  D(G(z)): 0.5249 / 0.5018
[1/50][6/782] Loss_D: 1.1383  Loss_G: 0.6907  D(x): 0.6589  D(G(z)): 0.5121 / 0.5013
[1/50][7/782] Loss_D: 1.1181  Loss_G: 0.6905  D(x): 0.6679  D(G(z)): 0.5093 / 0.5013
[1/50][8/782] Loss_D: 1.1024  Loss_G: 0.6900  D(x): 0.6764  D(G(z)): 0.5081 / 0.5016
[1/50][9/782] Loss_D: 1.0982  Loss_G: 0.6904  D(x): 0.6862  D(G(z)): 0.5132 / 0.5014
```

最后10行的输出数据

```
[50/50][772/782] Loss_D: 1.0064  Loss_G: 0.6931  D(x): 0.7311  D(G(z)): 0.5000 / 0.5000
[50/50][773/782] Loss_D: 1.0064  Loss_G: 0.6931  D(x): 0.7311  D(G(z)): 0.5000 / 0.5000
[50/50][774/782] Loss_D: 1.0064  Loss_G: 0.6931  D(x): 0.7311  D(G(z)): 0.5000 / 0.5000
[50/50][775/782] Loss_D: 1.0064  Loss_G: 0.6931  D(x): 0.7311  D(G(z)): 0.5000 / 0.5000
[50/50][776/782] Loss_D: 1.0064  Loss_G: 0.6931  D(x): 0.7311  D(G(z)): 0.5000 / 0.5000
[50/50][777/782] Loss_D: 1.0064  Loss_G: 0.6931  D(x): 0.7311  D(G(z)): 0.5000 / 0.5000
[50/50][778/782] Loss_D: 1.0064  Loss_G: 0.6931  D(x): 0.7311  D(G(z)): 0.5000 / 0.5000
[50/50][779/782] Loss_D: 1.0064  Loss_G: 0.6931  D(x): 0.7311  D(G(z)): 0.5000 / 0.5000
[50/50][780/782] Loss_D: 1.0064  Loss_G: 0.6931  D(x): 0.7311  D(G(z)): 0.5000 / 0.5000
[50/50][781/782] Loss_D: 1.0064  Loss_G: 0.6931  D(x): 0.7311  D(G(z)): 0.5000 / 0.5000
```

分数

```
Inception Score: 1.5730 ± 0.5228
```

