"""BiLSTM + Attention — ported from SARATHI-V2.

Architecture:
  - Input projection (n_features -> hidden*2) for residual connection
  - Bidirectional LSTM (n_features -> hidden, 2 layers)
  - LayerNorm on LSTM output + residual
  - Attention mechanism (Tanh + softmax weighted sum)
  - FC head with GELU activation -> single logit output
"""
import torch
import torch.nn as nn


class BiLSTM_Attention(nn.Module):
    def __init__(self, n_features=10, hidden=128, n_layers=2, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden * 2)
        self.lstm = nn.LSTM(
            n_features, hidden, n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.attn = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        res = self.input_proj(x)
        out, _ = self.lstm(x)
        out = self.norm(out + res)
        w = torch.softmax(self.attn(out), dim=1)
        ctx = (w * out).sum(dim=1)
        return self.fc(self.drop(ctx)).squeeze(-1)


class FatigueBiLSTM(nn.Module):
    """Original 8-feature BiLSTM (kept for reference)."""
    def __init__(self, inp=8, hs=128, nl=2, do=0.3):
        super().__init__()
        self.lstm = nn.LSTM(inp, hs, nl, batch_first=True, bidirectional=True, dropout=do)
        self.attn_layer = nn.Linear(hs * 2, 1)
        self.fc = nn.Sequential(nn.Linear(hs * 2, 64), nn.ReLU(), nn.Dropout(do), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn_layer(out), dim=1)
        ctx = (w * out).sum(dim=1)
        return self.fc(ctx).squeeze(-1)
