import torch
from torchvision.models import inception_v3, Inception_V3_Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True).to(device).eval()
x = torch.randn(1, 3, 299, 299).to(device)
with torch.no_grad():
    out = model(x)
print(type(out))
print(out)
