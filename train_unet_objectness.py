#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF


# ---------- UNet (doit matcher segment_unet.py) ----------
class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=64):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.mid = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))
        xm = self.mid(self.pool4(x4))

        y4 = self.up4(xm)
        y4 = self.conv4(torch.cat([y4, x4], dim=1))
        y3 = self.up3(y4)
        y3 = self.conv3(torch.cat([y3, x3], dim=1))
        y2 = self.up2(y3)
        y2 = self.conv2(torch.cat([y2, x2], dim=1))
        y1 = self.up1(y2)
        y1 = self.conv1(torch.cat([y1, x1], dim=1))
        return self.out(y1)


def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


class PairsDataset(Dataset):
    def __init__(self, pairs, size=384, augment=True):
        self.pairs = pairs
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rec = self.pairs[idx]
        ip = Path(rec["image_path"])
        mp = Path(rec["mask_path"])
        try:
            img = Image.open(ip).convert("RGB")
            msk = Image.open(mp).convert("L")
        except Exception:
            return None

        img = img.resize((self.size, self.size), resample=Image.BILINEAR)
        msk = msk.resize((self.size, self.size), resample=Image.NEAREST)

        x = TF.to_tensor(img)                # [0,1]
        y = TF.to_tensor(msk)
        y = (y > 0.5).float()                # binaire

        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = TF.hflip(x); y = TF.hflip(y)
        return x, y


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    x, y = zip(*batch)
    return torch.stack(x), torch.stack(y)


def load_pairs_jsonl(path: str):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, required=True, help="pairs_train.jsonl")
    ap.add_argument("--out_ckpt", type=str, default="checkpoints/unet_objectness.pt")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=4)   # 1060 -> 4 à 384 souvent OK
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs_jsonl(args.pairs)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pairs))

    n_val = int(len(pairs) * args.val_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    tr_pairs = [pairs[i] for i in tr_idx]
    va_pairs = [pairs[i] for i in val_idx]

    tr_ds = PairsDataset(tr_pairs, size=args.size, augment=True)
    va_ds = PairsDataset(va_pairs, size=args.size, augment=False)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, collate_fn=collate_skip_none)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, collate_fn=collate_skip_none)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print(f"Train pairs: {len(tr_pairs)} | Val pairs: {len(va_pairs)}")

    model = UNet(in_ch=3, out_ch=1, base=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    best_val = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for batch in tr_dl:
            if batch is None:
                continue
            x, y = batch
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = bce(logits, y) + dice_loss_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for batch in va_dl:
                if batch is None:
                    continue
                x, y = batch
                x = x.to(device); y = y.to(device)
                logits = model(x)
                loss = bce(logits, y) + dice_loss_with_logits(logits, y)
                va_losses.append(loss.item())

        tr_m = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_m = float(np.mean(va_losses)) if va_losses else 0.0
        print(f"Epoch {ep}/{args.epochs} | train={tr_m:.4f} | val={va_m:.4f}")

        if va_m < best_val:
            best_val = va_m
            torch.save({"state_dict": model.state_dict(), "size": args.size}, out_ckpt)
            print("Saved best:", out_ckpt)

    print("Done. Best val:", best_val)

if __name__ == "__main__":
    main()