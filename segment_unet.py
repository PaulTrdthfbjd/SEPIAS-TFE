#!/usr/bin/env python3
# segment_unet.py
import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from scipy.ndimage import label as cc_label, find_objects
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes

EXTS = ("*.jpg", "*.jpeg", "*.png")


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def collect_images(root: Path):
    paths = []
    for ext in EXTS:
        paths += glob(str(root / "**" / ext), recursive=True)
    return sorted(paths)


def apply_mask_rgba(crop_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # crop_rgb: uint8 HxWx3
    # mask: bool HxW (True = objet)
    alpha = (mask.astype(np.uint8) * 255)
    rgba = np.dstack([crop_rgb, alpha])  # HxWx4
    return rgba


def normalize_img(x: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------- Minimal UNet (binary) ----------
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="UNet checkpoint (.pt/.pth)")
    ap.add_argument("--ref_folder", type=str, default=None)
    ap.add_argument("--max_images", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="objects_unet")
    ap.add_argument("--manifest_path", type=str, default="objects_manifest_unet.jsonl")

    ap.add_argument("--input_size", type=int, default=512, help="Resize square for inference")
    ap.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for mask")
    ap.add_argument("--min_area", type=int, default=300)
    ap.add_argument("--max_objs_per_image", type=int, default=10)
    ap.add_argument("--max_area_ratio", type=float, default=0.6)

    args = ap.parse_args()

    load_dotenv(script_dir() / "test.env")
    ref_folder = Path(args.ref_folder).expanduser() if args.ref_folder else Path(os.getenv("REF_IMAGES_FOLDER", "")).expanduser()
    if not ref_folder.exists():
        raise FileNotFoundError(f"REF_IMAGES_FOLDER not found: {ref_folder}")

    out_dir = script_dir() / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = script_dir() / args.manifest_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = UNet(in_ch=3, out_ch=1, base=64).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    img_paths = collect_images(ref_folder)

    if args.max_images > 0:
        img_paths = img_paths[: min(args.max_images, len(img_paths))]

    print(f"Found {len(img_paths)} images in {ref_folder}")

    n_written = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for idx, p in enumerate(img_paths, start=1):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue

            img_np = np.array(img)
            H, W = img_np.shape[:2]

            # preprocess
            x = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # (3,H,W)
            x = x.unsqueeze(0).to(device)
            #x = normalize_img(x)

            # resize
            x_rs = F.interpolate(x, size=(args.input_size, args.input_size), mode="bilinear", align_corners=False)

            with torch.no_grad():
                logits = model(x_rs)  # (1,1,h,w)
                prob = torch.sigmoid(logits)

            # back to original
            prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            mask = (prob.detach().cpu().numpy() >= args.threshold)
            mask = binary_closing(mask, iterations=2)   # connecte les îlots
            mask = binary_opening(mask, iterations=1)   # enlève le bruit fin
            mask = binary_fill_holes(mask)              # remplit les trous internes

            area = int(mask.sum())
            if area == 0:
                continue
            if (area / float(H * W)) > args.max_area_ratio:
                # mask covers almost all image -> not useful as object crop
                continue

            # connected components -> multiple crops
            labeled, n_cc = cc_label(mask)
            if n_cc == 0:
                continue

            slices = find_objects(labeled)
            # sort components by area desc
            comps = []
            for cc_id, sl in enumerate(slices, start=1):
                if sl is None:
                    continue
                cc_mask = (labeled[sl] == cc_id)
                a = int(cc_mask.sum())
                comps.append((a, cc_id, sl))
            comps.sort(reverse=True, key=lambda t: t[0])
            comps = comps[: args.max_objs_per_image]

            for j, (a, cc_id, sl) in enumerate(comps):
                if a < args.min_area:
                    continue
                ysl, xsl = sl
                y1, y2 = ysl.start, ysl.stop - 1
                x1, x2 = xsl.start, xsl.stop - 1

                crop = img_np[y1:y2 + 1, x1:x2 + 1]
                crop_mask = (labeled[y1:y2 + 1, x1:x2 + 1] == cc_id)
                crop_out = apply_mask_rgba(crop, crop_mask)

                obj_id = f"{idx:06d}_{j:02d}"
                crop_path = out_dir / f"{obj_id}_person.png"
                Image.fromarray(crop, mode="RGB").save(crop_path)

                record = {
                    "obj_id": obj_id,
                    "label_name": "object",
                    "score": None,
                    "parent_path": os.path.normpath(str(p)),
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "crop_path": os.path.normpath(str(crop_path)),
                    "method": "unet",
                    "checkpoint": os.path.normpath(args.checkpoint),
                }
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(img_paths)} | crops: {n_written}")

    print("Saved manifest:", manifest_path)
    print("Total crops saved:", n_written)
    print("Out dir:", out_dir)


if __name__ == "__main__":
    main()