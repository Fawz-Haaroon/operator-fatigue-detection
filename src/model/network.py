"""BiLSTM + Attention."""
import torch, torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hs): super().__init__(); self.attn = nn.Linear(hs * 2, 1)
    def forward(self, out):
        w = torch.softmax(self.attn(out), dim=1)
        return torch.sum(w * out, dim=1)

class FatigueBiLSTM(nn.Module):
    def __init__(self, inp=8, hs=128, nl=2, do=0.3):
        super().__init__()
        self.lstm = nn.LSTM(inp, hs, nl, batch_first=True, bidirectional=True, dropout=do)
        self.attn = Attention(hs)
        self.fc = nn.Sequential(nn.Linear(hs*2, 64), nn.ReLU(), nn.Dropout(do), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.attn(out)).squeeze(-1)