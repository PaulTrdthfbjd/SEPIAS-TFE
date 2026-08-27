#!/usr/bin/env python3
# segment_segnet.py
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


# ---------- Minimal SegNet (binary) ----------
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
        self.dec4 = conv_block(512, 256, n=3)   # <-- IMPORTANT (512 -> 256)

        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.dec3 = conv_block(256, 128, n=3)   # <-- IMPORTANT (256 -> 128)

        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dec2 = conv_block(128, 64, n=2)    # <-- IMPORTANT (128 -> 64)

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
        x = self.unpool4(x, i4, output_size=s4); x = self.dec4(x)  # sort 256
        x = self.unpool3(x, i3, output_size=s3); x = self.dec3(x)  # sort 128
        x = self.unpool2(x, i2, output_size=s2); x = self.dec2(x)  # sort 64
        x = self.unpool1(x, i1, output_size=s1); x = self.dec1(x)

        return self.out(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="SegNet checkpoint (.pt/.pth)")
    ap.add_argument("--ref_folder", type=str, default=None)
    ap.add_argument("--max_images", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="objects_segnet")
    ap.add_argument("--manifest_path", type=str, default="objects_manifest_segnet.jsonl")

    ap.add_argument("--input_size", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.5)
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

    model = SegNetBinary(in_ch=3, out_ch=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    infer_size = ckpt.get("size", 256) if isinstance(ckpt, dict) else 256
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

            x = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            x = x.unsqueeze(0).to(device)
            #x = normalize_img(x)

            x_rs = F.interpolate(x, size=(args.input_size, args.input_size), mode="bilinear", align_corners=False)

            with torch.no_grad():
                logits = model(x_rs)
                prob = torch.sigmoid(logits)

            prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            mask = (prob.detach().cpu().numpy() >= args.threshold)
            mask = binary_closing(mask, iterations=2)   # connecte les îlots
            mask = binary_opening(mask, iterations=1)   # enlève le bruit fin
            mask = binary_fill_holes(mask)              # remplit les trous internes

            area = int(mask.sum())
            if area == 0:
                continue
            if (area / float(H * W)) > args.max_area_ratio:
                continue

            labeled, n_cc = cc_label(mask)
            if n_cc == 0:
                continue

            slices = find_objects(labeled)
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
                    "method": "segnet",
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