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


def apply_mask_rgba(crop_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # crop_rgb: uint8 HxWx3
    # mask: bool HxW (True = objet)
    alpha = (mask.astype(np.uint8) * 255)
    rgba = np.dstack([crop_rgb, alpha])  # HxWx4
    return rgba


def xywh_to_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w - 1), int(y + h - 1)
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_folder", type=str, default=None, help="Override REF_IMAGES_FOLDER")
    ap.add_argument("--max_images", type=int, default=200)
    ap.add_argument("--out_dir", type=str, default="objects_sam_auto")
    ap.add_argument("--manifest_path", type=str, default="objects_manifest_sam_auto.jsonl")

    # SAM checkpoint + model type
    ap.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM .pth checkpoint")
    ap.add_argument("--model_type", type=str, default="vit_b", choices=["vit_h", "vit_l", "vit_b"])

    # mask generator params (trade-off speed/quality)
    ap.add_argument("--points_per_side", type=int, default=16)
    ap.add_argument("--pred_iou_thresh", type=float, default=0.86)
    ap.add_argument("--stability_score_thresh", type=float, default=0.92)
    ap.add_argument("--crop_n_layers", type=int, default=0)

    # filtering
    ap.add_argument("--max_masks_per_image", type=int, default=15)
    ap.add_argument("--min_area", type=int, default=400)
    ap.add_argument("--max_area_ratio", type=float, default=0.65)

    args = ap.parse_args()

    load_dotenv(script_dir() / "test.env")

    ref_folder = Path(args.ref_folder).expanduser() if args.ref_folder else Path(os.getenv("REF_IMAGES_FOLDER", "")).expanduser()
    if not ref_folder.exists():
        raise FileNotFoundError(f"REF_IMAGES_FOLDER not found: {ref_folder}")

    out_dir = script_dir() / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = script_dir() / args.manifest_path

    # SAM import
    try:
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except Exception as e:
        raise ImportError(
            "segment-anything n'est pas installé.\n"
            "Installe-le (voir commandes dans ma réponse) puis relance.\n"
            f"Erreur: {e}"
        )

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

    img_paths = collect_images(ref_folder)
    if args.max_images > 0:
        img_paths = img_paths[: args.max_images]
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

            anns = mask_generator.generate(img_np)  # list of dicts
            if not anns:
                continue

            # sort: larger / higher quality first (you can change to predicted_iou)
            anns.sort(key=lambda a: float(a.get("predicted_iou", 0.0)), reverse=True)

            kept = 0
            for j, ann in enumerate(anns):
                if kept >= args.max_masks_per_image:
                    break
                mask = ann["segmentation"].astype(bool)
                area = int(mask.sum())
                if area < args.min_area:
                    continue
                if (area / float(H * W)) > args.max_area_ratio:
                    continue

                x1, y1, x2, y2 = xywh_to_xyxy(ann["bbox"])
                # pad
                pad = 5
                x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)

                crop = img_np[y1:y2 + 1, x1:x2 + 1]
                crop_mask = mask[y1:y2 + 1, x1:x2 + 1]
                crop_out = apply_mask_rgba(crop, crop_mask)

                obj_id = f"{idx:06d}_{kept:02d}"
                crop_path = out_dir / f"{obj_id}_obj.png"
                Image.fromarray(crop, mode="RGB").save(crop_path)

                record = {
                    "obj_id": obj_id,
                    "label_name": "object",
                    "score": float(ann.get("predicted_iou", 0.0)),
                    "parent_path": os.path.normpath(str(p)),
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "crop_path": os.path.normpath(str(crop_path)),
                    "method": "sam_auto",
                    "model_type": args.model_type,
                }
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1
                kept += 1

            if idx % 25 == 0:
                print(f"Processed {idx}/{len(img_paths)} | crops: {n_written}")

    print("Saved manifest:", manifest_path)
    print("Total crops saved:", n_written)
    print("Out dir:", out_dir)


if __name__ == "__main__":
    main()