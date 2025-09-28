import os
import re
import time
import collections
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter

# Utils: Animator / Timer / Accumulator
# 训练可视化
class TBAnimator:
    def __init__(self, metric_names=('loss',), log_dir='logs'):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.metric_names = metric_names
        self.history = []

    def add(self, step, metrics):
        self.history.append((step, *metrics))
        # 写入 TensorBoard
        for name, val in zip(self.metric_names, metrics):
            self.writer.add_scalar(name, float(val), global_step=step)
        # 同时在控制台打印一行
        printable = ", ".join(f"{n}={v:.4f}" for n, v in zip(self.metric_names, metrics))
        print(f"epoch {step}: {printable}")

    def close(self):
        self.writer.close()

# 通过计算 token总数 ÷ Timer测到的秒数 来获取 tokens/sec
class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
    def stop(self):
        return time.perf_counter() - self.t0

# 累加器
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def __getitem__(self, idx):
        return self.data[idx]

# 数据预处理
class Vocab:
    def __init__(self, tokens, min_freq=2,
                 reserved_tokens=('<pad>', '<bos>', '<eos>', '<unk>')):
        counter = collections.Counter([t for line in tokens for t in line])
        self.idx_to_token = list(reserved_tokens)
        self.token_to_idx = {t:i for i,t in enumerate(self.idx_to_token)}
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in self.token_to_idx:
                self.token_to_idx[tok] = len(self.idx_to_token)
                self.idx_to_token.append(tok)
    def __len__(self): return len(self.idx_to_token)
    def __getitem__(self, token):
        if isinstance(token, (list, tuple)):
            return [self.__getitem__(t) for t in token]
        return self.token_to_idx.get(token, self.token_to_idx['<unk>'])
    def to_tokens(self, idx):
        if isinstance(idx, (list, tuple)):
            return [self.to_tokens(i) for i in idx]
        return self.idx_to_token[idx]

def tokenize(s):
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s.split(' ') if s else []

def truncate_pad(tokens, num_steps, pad_token):
    tokens = tokens[:num_steps]
    length = min(len(tokens), num_steps)
    return tokens + [pad_token] * (num_steps - len(tokens)), length

def build_arrays(src_lines, tgt_lines, num_steps, src_vocab, tgt_vocab):
    # Y 不包含 <bos>，只在训练时拼接 bos + Y[:, :-1]
    src_idxs, src_lens, tgt_idxs, tgt_lens = [], [], [], []
    for s, t in zip(src_lines, tgt_lines):
        s_tok = tokenize(s)
        t_tok = tokenize(t) + ['<eos>']
        s_tok, s_len = truncate_pad(s_tok, num_steps, '<pad>')
        t_tok, t_len = truncate_pad(t_tok, num_steps, '<pad>')
        src_idxs.append(src_vocab[s_tok]); src_lens.append(s_len)
        tgt_idxs.append(tgt_vocab[t_tok]); tgt_lens.append(t_len)
    X = torch.tensor(src_idxs, dtype=torch.long)
    X_len = torch.tensor(src_lens, dtype=torch.long)
    Y = torch.tensor(tgt_idxs, dtype=torch.long)
    Y_len = torch.tensor(tgt_lens, dtype=torch.long)
    return X, X_len, Y, Y_len

def load_data_nmt_local(batch_size, num_steps, path='fra.txt', num_examples=10000, direction=(0,1)):
    """
    读取本地 'fra.txt'（或你自己的并行语料）:
    每行形如: 'go .\tje vais !'
    direction=(src_col, tgt_col) 选择列方向。
    返回: train_iter, src_vocab, tgt_vocab
    """
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('\t') for line in f if '\t' in line]
        src_col, tgt_col = direction
        pairs = [(l[src_col], l[tgt_col]) for l in lines[:num_examples]]
    else:
        # 小样本内置示例（英文->法文）
        pairs = [
            ("go .", "va !"),
            ("i love you .", "je t aime ."),
            ("he is a student .", "il est etudiant ."),
            ("she is reading a book .", "elle lit un livre ."),
            ("we like machine learning .", "nous aimons l apprentissage automatique ."),
        ]

    src_tokens = [tokenize(s) for s,_ in pairs]
    tgt_tokens = [tokenize(t) for _,t in pairs]

    src_vocab = Vocab(src_tokens, min_freq=1)
    tgt_vocab = Vocab([['<eos>']] + tgt_tokens, min_freq=1)

    X, X_len, Y, Y_len = build_arrays([s for s,_ in pairs], [t for _,t in pairs],
                                      num_steps, src_vocab, tgt_vocab)
    dataset = TensorDataset(X, X_len, Y, Y_len)
    train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_iter, src_vocab, tgt_vocab

# Seq2Seq Model
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError
    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        dec_output, dec_state = self.decoder(dec_X, dec_state)
        return dec_output, dec_state

class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
    def forward(self, X, *args):
        # X: (batch, seq_len)
        X = self.embedding(X)            # (batch, seq_len, embed)
        X = X.permute(1, 0, 2)           # (seq_len, batch, embed)
        output, state = self.rnn(X)      # output: (seq_len, batch, hidden)
        return output, state             # state: (num_layers, batch, hidden)

class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
    def init_state(self, enc_outputs, *args):
        # enc_outputs = (enc_outputs_all_steps, enc_state)
        return enc_outputs[1]  # 仅用最终隐状态初始化解码器
    def forward(self, X, state):
        # X: (batch, seq_len)
        X = self.embedding(X).permute(1, 0, 2)   # (seq_len, batch, embed)
        context = state[-1].repeat(X.shape[0], 1, 1)  # (seq_len, batch, hidden)
        X_and_context = torch.cat((X, context), dim=2)
        output, state = self.rnn(X_and_context, state)    # output: (seq_len, batch, hidden)
        output = self.dense(output).permute(1, 0, 2)      # (batch, seq_len, vocab)
        return output, state

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len=None, *args, **kwargs):
        # pred: (batch, seq_len, vocab), label: (batch, seq_len)
        weights = torch.ones_like(label)
        if valid_len is not None:
            weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted = super().forward(pred.permute(0, 2, 1), label)  # (batch, seq_len)
        weighted = (unweighted * weights).mean(dim=1)               # (batch,)
        return weighted

# =========================
# Train
# =========================
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.GRU):
            for name, p in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(p)

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = MaskedSoftmaxCELoss().to(device)
    net.train()

    animator = TBAnimator(metric_names=('loss',), log_dir='logs')
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # total_loss, num_tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            # Teacher forcing: dec_input = <bos> + Y[:-1]
            bos_id = tgt_vocab['<bos>']
            bos = torch.full((Y.shape[0], 1), fill_value=bos_id, dtype=torch.long, device=device)
            dec_input = torch.cat([bos, Y[:, :-1]], dim=1)

            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss_fn(Y_hat, Y, Y_valid_len)  # (batch,)
            l.sum().backward()

            nn_utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            num_tokens = Y_valid_len.sum()
            optimizer.step()

            with torch.no_grad():
                metric.add(l.sum(), num_tokens)

        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))

    elapsed = timer.stop()
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / elapsed:.1f} tokens/sec on {str(device)}')

if __name__ == "__main__":
    # 参数
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 50  # 训练轮数先设少一点试跑
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据
    train_iter, src_vocab, tgt_vocab = load_data_nmt_local(
        batch_size=batch_size, num_steps=num_steps, path='fra.txt'
    )

    # 模型
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout).to(device)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout).to(device)
    net = EncoderDecoder(encoder, decoder).to(device)
    # 训练
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
