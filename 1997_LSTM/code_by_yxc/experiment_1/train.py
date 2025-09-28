import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Lstm
from torch.utils.tensorboard import SummaryWriter
VOCAB = ['B','T','P','S','X','V','E']
stoi = {c:i for i,c in enumerate(VOCAB)}
num_classes = 2
def encode_seq(s):
    idxs = [stoi[c] for c in s]
    onehots = torch.zeros(len(s), len(VOCAB))
    onehots[torch.arange(len(s)), idxs] = 1.0
    return onehots
class ReberDataset(Dataset):
    def __init__(self, legal_txt, illegal_txt):
        self.data = []
        for line in open(legal_txt, 'r', encoding='utf-8'):
            s = line.strip()
            if s: self.data.append((encode_seq(s), 1))
        for line in open(illegal_txt, 'r', encoding='utf-8'):
            s = line.strip()
            if s: self.data.append((encode_seq(s), 0))
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        x, y = self.data[i]
        return x, len(x), y
def collate_fn(batch):
    batch.sort(key=lambda t: t[1], reverse=True)
    xs, lens, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    lens = torch.tensor(lens, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, lens, ys
if __name__ == "__main__":
    writer = SummaryWriter("logs")
    device = torch.device("cuda")
    legal_txt   = r".\train\10000_legal_ReberGrammar.txt"
    illegal_txt = r".\train\10000_illegal_ReberGrammar.txt"
    full = ReberDataset(legal_txt, illegal_txt)
    n = len(full)
    idx = int(n*0.8)
    train_ds, val_ds = torch.utils.data.random_split(full, [idx, n-idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    lstm = Lstm().to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    learning_rate = 0.001
    optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
    lstm.train()
    epoch = 15
    for i in range(epoch):
        print("第", i + 1, "轮")
        train_loss = 0.0
        for x, lens, y in train_loader:
            optimizer.zero_grad()
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            logits = lstm(x, lens)
            loss = loss_function(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=i + 1)
        print(train_loss)
        torch.save(lstm.state_dict(), "lstm_model_{}.pth".format(i + 1))
    writer.close()