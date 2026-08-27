#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from dotenv import load_dotenv

EXTS = ("*.jpg", "*.jpeg", "*.png")

def script_dir() -> Path:
    return Path(__file__).resolve().parent

def collect_images(root: Path):
    paths = []
    for ext in EXTS:
        paths += glob(str(root / "**" / ext), recursive=True)
    return sorted(paths)

def safe_rel_name(p: Path, root: Path) -> str:
    rel = p.relative_to(root)
    s = str(rel).replace("\\", "__").replace("/", "__")
    # évite les collisions sur différents dossiers
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_folder", type=str, default=None, help="Override REF_IMAGES_FOLDER")
    ap.add_argument("--out_masks", type=str, required=True, help="Output masks folder")
    ap.add_argument("--out_pairs", type=str, required=True, help="Output pairs jsonl")
    ap.add_argument("--max_images", type=int, default=500)

    ap.add_argument("--sam_checkpoint", type=str, required=True)
    ap.add_argument("--model_type", type=str, default="vit_b", choices=["vit_h", "vit_l", "vit_b"])

    ap.add_argument("--points_per_side", type=int, default=16)
    ap.add_argument("--pred_iou_thresh", type=float, default=0.86)
    ap.add_argument("--stability_score_thresh", type=float, default=0.92)
    ap.add_argument("--crop_n_layers", type=int, default=0)

    ap.add_argument("--top_masks", type=int, default=20)
    ap.add_argument("--min_area", type=int, default=400)
    ap.add_argument("--max_area_ratio", type=float, default=0.85)
    args = ap.parse_args()

    load_dotenv(script_dir() / "test.env")

    ref_folder = Path(args.ref_folder).expanduser() if args.ref_folder else Path(os.getenv("REF_IMAGES_FOLDER", "")).expanduser()
    if not ref_folder.exists():
        raise FileNotFoundError(f"REF_IMAGES_FOLDER not found: {ref_folder}")

    out_masks = script_dir() / args.out_masks
    out_masks.mkdir(parents=True, exist_ok=True)
    out_pairs = script_dir() / args.out_pairs

    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        output_mode="binary_mask",
    )

    img_paths = collect_images(ref_folder)[: args.max_images]
    print(f"Found {len(img_paths)} images in {ref_folder}")

    n_saved = 0
    with open(out_pairs, "w", encoding="utf-8") as f_pairs:
        for i, p in enumerate(img_paths, start=1):
            p = Path(p)
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue

            img_np = np.array(img)
            H, W = img_np.shape[:2]

            anns = mask_generator.generate(img_np)
            if not anns:
                continue

            anns.sort(key=lambda a: float(a.get("predicted_iou", 0.0)), reverse=True)

            union = np.zeros((H, W), dtype=bool)
            kept = 0
            for ann in anns:
                if kept >= args.top_masks:
                    break
                m = ann["segmentation"].astype(bool)
                area = int(m.sum())
                if area < args.min_area:
                    continue
                if (area / float(H * W)) > args.max_area_ratio:
                    continue
                union |= m
                kept += 1

            if union.sum() == 0:
                continue

            mask_name = safe_rel_name(p, ref_folder) + ".png"
            mask_path = out_masks / mask_name
            Image.fromarray((union.astype(np.uint8) * 255)).save(mask_path)

            rec = {
                "image_path": os.path.normpath(str(p)),
                "mask_path": os.path.normpath(str(mask_path)),
                "source": "sam_union",
                "model_type": args.model_type
            }
            f_pairs.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_saved += 1

            if i % 50 == 0:
                print(f"Processed {i}/{len(img_paths)} | masks saved: {n_saved}")

    print("Saved masks to:", out_masks)
    print("Saved pairs to:", out_pairs)
    print("Total pairs:", n_saved)

if __name__ == "__main__":
    main()