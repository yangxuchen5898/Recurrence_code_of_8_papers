import torch
from torch import nn
from torch.nn import LSTM, Linear
from torch.nn.utils.rnn import pack_padded_sequence
class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.lstm = LSTM(
            input_size = 7,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True,
            bidirectional = False
        )
        self.fc = Linear(64, 2)
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first = True, enforce_sorted = False)
        _, (h_n, c_n) = self.lstm(packed)
        last = h_n[-1]
        logits = self.fc(last)
        return logits
if __name__ == '__main__':
    lstm = Lstm()
    if torch.cuda.is_available():
        lstm.cuda()
    print(lstm)