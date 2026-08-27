#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF


def conv_block(c_in, c_out, n=2):
    layers = []
    for i in range(n):
        layers += [
            nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class SegNetBinary(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.enc1 = conv_block(in_ch, 64, n=2)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc2 = conv_block(64, 128, n=2)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc3 = conv_block(128, 256, n=3)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc4 = conv_block(256, 512, n=3)
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc5 = conv_block(512, 512, n=3)
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)

        self.unpool5 = nn.MaxUnpool2d(2, 2)
        self.dec5 = conv_block(512, 512, n=3)

        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.dec4 = conv_block(512, 256, n=3)   # <-- 512 -> 256 (important)

        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.dec3 = conv_block(256, 128, n=3)   # <-- 256 -> 128

        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dec2 = conv_block(128, 64, n=2)    # <-- 128 -> 64

        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.dec1 = conv_block(64, 64, n=2)

        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = self.enc1(x); s1 = x1.size()
        x, i1 = self.pool1(x1)

        x2 = self.enc2(x); s2 = x2.size()
        x, i2 = self.pool2(x2)

        x3 = self.enc3(x); s3 = x3.size()
        x, i3 = self.pool3(x3)

        x4 = self.enc4(x); s4 = x4.size()
        x, i4 = self.pool4(x4)

        x5 = self.enc5(x); s5 = x5.size()
        x, i5 = self.pool5(x5)

        x = self.unpool5(x, i5, output_size=s5); x = self.dec5(x)
        x = self.unpool4(x, i4, output_size=s4); x = self.dec4(x)   # <-- sort 256
        x = self.unpool3(x, i3, output_size=s3); x = self.dec3(x)   # <-- sort 128
        x = self.unpool2(x, i2, output_size=s2); x = self.dec2(x)   # <-- sort 64
        x = self.unpool1(x, i1, output_size=s1); x = self.dec1(x)

        return self.out(x)


def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


class PairsDataset(Dataset):
    def __init__(self, pairs, size=256, augment=True):
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

        x = TF.to_tensor(img)
        y = TF.to_tensor(msk)
        y = (y > 0.5).float()

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
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--out_ckpt", type=str, default="checkpoints/segnet_objectness.pt")
    ap.add_argument("--size", type=int, default=256)      # SegNet lourd -> 256 recommandé sur 1060
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
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

    model = SegNetBinary(in_ch=3, out_ch=1).to(device)
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