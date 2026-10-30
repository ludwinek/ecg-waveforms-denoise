"""1D U-Net denoiser (PyTorch). Imported lazily - the rest of the package does not need torch."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU(),
            nn.Conv1d(cout, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    """Window length must be divisible by 8 (720 samples = 2 s at 360 Hz).
    The network predicts the NOISE; the output is input - noise_estimate (= clean-ECG estimate)."""

    def __init__(self, base=16, k=9):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3 = ConvBlock(1, c[0], k), ConvBlock(c[0], c[1], k), ConvBlock(c[1], c[2], k)
        self.b = ConvBlock(c[2], c[3], k)
        self.u3, self.d3 = nn.ConvTranspose1d(c[3], c[2], 2, stride=2), ConvBlock(c[2] * 2, c[2], k)
        self.u2, self.d2 = nn.ConvTranspose1d(c[2], c[1], 2, stride=2), ConvBlock(c[1] * 2, c[1], k)
        self.u1, self.d1 = nn.ConvTranspose1d(c[1], c[0], 2, stride=2), ConvBlock(c[0] * 2, c[0], k)
        self.head = nn.Conv1d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool1d(e1, 2))
        e3 = self.e3(F.max_pool1d(e2, 2))
        b = self.b(F.max_pool1d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return x - self.head(d1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def _val_loss(model, vx, vy, device, chunk=1024):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(vx), chunk):
            x = torch.from_numpy(vx[i:i + chunk]).to(device)
            y = torch.from_numpy(vy[i:i + chunk]).to(device)
            tot += F.mse_loss(model(x), y, reduction="sum").item()
            n += y.numel()
    return tot / n


def train_model(sampler, val_x, val_y, device, epochs=25, steps=200, bs=256, lr=2e-3, patience=6,
                base=16, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = UNet1D(base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for _ in range(steps):
            x, y = sampler.batch(bs, rng)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
        vl = _val_loss(model, val_x, val_y, device)
        log(f"    epoch {ep + 1:2d}/{epochs}  train {tl / steps:.4f}  val {vl:.4f}")
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log("    early stop")
                break
    model.load_state_dict(best_state)
    return model, best


def make_predictor(model, device, chunk=512):
    model.eval()

    def fn(x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), chunk):
                out.append(model(torch.from_numpy(x[i:i + chunk]).to(device)).cpu().numpy())
        return np.concatenate(out)

    return fn
