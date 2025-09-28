import torch
from torch.utils.data import DataLoader
from model import Lstm
from train import ReberDataset, collate_fn


device = torch.device("cuda")
lstm = Lstm()
legal_txt   = r".\test\500_legal_ReberGrammar.txt"
illegal_txt = r".\test\500_illegal_ReberGrammar.txt"
test_dataset = ReberDataset(legal_txt, illegal_txt)
n = len(test_dataset)
idx = int(n*0.8)
test_ds, val_ds = torch.utils.data.random_split(test_dataset, [idx, n-idx])
test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
lstm.to(device).eval()
state = torch.load("lstm_model_15.pth", map_location=device)
msg = lstm.load_state_dict(state, strict=True)
with torch.no_grad():
    total, correct = 0, 0
    for x, lens, y in test_loader:
        x, lens, y = x.to(device), lens.to(device), y.to(device)
        logits = lstm(x, lens)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
acc = correct/total
print(acc)