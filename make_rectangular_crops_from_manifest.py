#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from PIL import Image


def normpath(p: str) -> str:
    return os.path.normpath(str(p))


def clamp_bbox(bbox, width: int, height: int, pad: int = 0):
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width - 1, x2 + pad)
    y2 = min(height - 1, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def safe_stem(path: str) -> str:
    p = Path(path)
    return p.stem.replace(" ", "_").replace("/", "_").replace("\\", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Input manifest jsonl containing parent_path and bbox_xyxy")
    ap.add_argument("--out_dir", required=True, help="Output directory for rectangular crops")
    ap.add_argument("--out_manifest", required=True, help="Output manifest jsonl for rectangular crops")
    ap.add_argument("--pad", type=int, default=5, help="Padding added around bbox")
    ap.add_argument("--ext", choices=["jpg", "png"], default="jpg", help="Output image format")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_manifest = Path(args.out_manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    n_in = 0
    n_out = 0
    n_skip = 0

    with open(manifest_path, "r", encoding="utf-8") as f_in, \
         open(out_manifest, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            n_in += 1

            try:
                rec = json.loads(line)
            except Exception:
                n_skip += 1
                continue

            parent_path = rec.get("parent_path")
            bbox = rec.get("bbox_xyxy")

            if not parent_path or bbox is None:
                n_skip += 1
                continue

            if not os.path.exists(parent_path):
                n_skip += 1
                continue

            try:
                parent_img = Image.open(parent_path).convert("RGB")
            except Exception:
                n_skip += 1
                continue

            W, H = parent_img.size
            bb = clamp_bbox(bbox, W, H, pad=args.pad)
            if bb is None:
                n_skip += 1
                continue

            x1, y1, x2, y2 = bb

            # PIL crop uses exclusive right/lower bounds.
            crop = parent_img.crop((x1, y1, x2 + 1, y2 + 1)).convert("RGB")

            old_crop_path = rec.get("crop_path", f"crop_{n_in:06d}")
            filename = f"{n_out:06d}_{safe_stem(old_crop_path)}.{args.ext}"
            crop_path = out_dir / filename

            if args.ext == "jpg":
                crop.save(crop_path, quality=args.quality, optimize=True)
            else:
                crop.save(crop_path)

            new_rec = dict(rec)
            new_rec["source_crop_path"] = normpath(rec.get("crop_path", ""))
            new_rec["crop_path"] = normpath(str(crop_path))
            new_rec["bbox_xyxy"] = [int(x1), int(y1), int(x2), int(y2)]
            new_rec["crop_mode"] = "rectangular_bbox_rgb"
            new_rec["crop_ext"] = args.ext
            new_rec["pad"] = int(args.pad)

            f_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Input records: {n_in}")
    print(f"Written rectangular crops: {n_out}")
    print(f"Skipped: {n_skip}")
    print(f"Output crops: {out_dir.resolve()}")
    print(f"Output manifest: {out_manifest.resolve()}")


if __name__ == "__main__":
    main()