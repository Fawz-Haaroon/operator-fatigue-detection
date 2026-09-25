"""Training script."""
import argparse, torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.model.network import FatigueBiLSTM

def train(args):
    X = torch.tensor(np.load(f"{args.data}/features.npy"), dtype=torch.float32)
    y = torch.tensor(np.load(f"{args.data}/labels.npy"), dtype=torch.float32)
    s = int(0.8 * len(X))
    tl = DataLoader(TensorDataset(X[:s], y[:s]), batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(TensorDataset(X[s:], y[s:]), batch_size=args.batch_size)
    m = FatigueBiLSTM()
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    crit = nn.BCELoss()
    best = float("inf")
    for e in range(args.epochs):
        m.train(); tl_loss = sum(((opt.zero_grad(), crit(m(xb), yb).backward(), opt.step(), crit(m(xb), yb).item())[-1] for xb, yb in tl)) / len(tl)
        m.eval()
        with torch.no_grad(): vl_loss = sum(crit(m(xb), yb).item() for xb, yb in vl) / len(vl)
        print(f"Epoch {e+1}/{args.epochs} — train: {tl_loss:.4f}, val: {vl_loss:.4f}")
        if vl_loss < best: best = vl_loss; torch.save(m.state_dict(), args.output); print(f"  Saved to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--output", default="models/fatigue_bilstm.pth")
    train(p.parse_args())