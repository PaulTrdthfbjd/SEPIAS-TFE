#!/usr/bin/env python3
# segment_sam3.py
import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from dotenv import load_dotenv

# ---------- Helpers ----------
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


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())  # x1,y1,x2,y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_folder", type=str, default=None, help="Override REF_IMAGES_FOLDER")
    ap.add_argument("--max_images", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="objects_sam3")
    ap.add_argument("--manifest_path", type=str, default="objects_manifest_sam3.jsonl")

    ap.add_argument("--concept", type=str, default="person", help="Text concept prompt (e.g., person, horse, church)")
    ap.add_argument("--model_id", type=str, default="facebook/sam3",
                    help="SAM3 model id (may be gated). You can use an ungated mirror if needed.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Instance filtering threshold (post_process_instance_segmentation)")
    ap.add_argument("--mask_threshold", type=float, default=0.5,
                    help="Mask binarization threshold (post_process_instance_segmentation)")

    ap.add_argument("--max_objs_per_image", type=int, default=10)
    ap.add_argument("--min_area", type=int, default=300)
    ap.add_argument("--max_area_ratio", type=float, default=0.6, help="Skip masks larger than this ratio of image area")

    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    load_dotenv(script_dir() / "test.env")

    ref_folder = Path(args.ref_folder).expanduser() if args.ref_folder else Path(os.getenv("REF_IMAGES_FOLDER", "")).expanduser()
    if not ref_folder.exists():
        raise FileNotFoundError(f"REF_IMAGES_FOLDER not found: {ref_folder}")

    out_dir = script_dir() / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = script_dir() / args.manifest_path

    # ---- Import SAM3 (Transformers) ----
    try:
        from transformers import Sam3Processor, Sam3Model
    except Exception as e:
        raise ImportError(
            "Impossible d'importer Sam3Processor/Sam3Model.\n"
            "Ton transformers ne contient probablement pas SAM3.\n"
            "Solution: upgrade transformers (et/ou utilise un autre env) puis réessaie.\n"
            f"Erreur: {e}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = Sam3Processor.from_pretrained(args.model_id)
    model = Sam3Model.from_pretrained(args.model_id).to(device)
    model.eval()

    img_paths = collect_images(ref_folder)[: args.max_images]
    print(f"Found {len(img_paths)} images in {ref_folder}")

    n_written = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        # batch by batch
        for start in range(0, len(img_paths), args.batch_size):
            batch_paths = img_paths[start:start + args.batch_size]
            images = []
            orig_sizes = []
            valid_paths = []

            for p in batch_paths:
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:
                    continue
                images.append(im)
                orig_sizes.append(im.size[::-1])  # (H,W)
                valid_paths.append(p)

            if len(images) == 0:
                continue

            texts = [args.concept] * len(images)
            inputs = processor(images=images, text=texts, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )

            # results is list of dicts, one per image
            for local_i, res in enumerate(results):
                parent_path = str(valid_paths[local_i])
                img = images[local_i]
                img_np = np.array(img)
                H, W = img_np.shape[:2]

                masks = res.get("masks", None)
                scores = res.get("scores", None)
                if masks is None or len(masks) == 0:
                    continue

                masks = masks.detach().cpu().numpy().astype(bool)
                if scores is not None:
                    scores = scores.detach().cpu().numpy()
                else:
                    scores = np.ones((masks.shape[0],), dtype=np.float32)

                # sort by score desc
                order = np.argsort(-scores)[: args.max_objs_per_image]

                for j_idx, j in enumerate(order):
                    m = masks[j]
                    area = int(m.sum())
                    if area < args.min_area:
                        continue
                    if (area / float(H * W)) > args.max_area_ratio:
                        continue

                    bb = bbox_from_mask(m)
                    if bb is None:
                        continue
                    x1, y1, x2, y2 = bb

                    # small padding
                    pad = 5
                    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                    x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)

                    crop = img_np[y1:y2 + 1, x1:x2 + 1]
                    crop_mask = m[y1:y2 + 1, x1:x2 + 1]
                    crop_out = apply_mask_rgba(crop, crop_mask)

                    obj_id = f"{start + local_i:06d}_{j_idx:02d}"
                    crop_path = out_dir / f"{obj_id}_{args.concept}.png"
                    Image.fromarray(crop, mode="RGB").save(crop_path)

                    record = {
                        "obj_id": obj_id,
                        "label_name": args.concept,
                        "score": float(scores[j]),
                        "parent_path": parent_path,
                        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                        "crop_path": str(crop_path),
                        "method": "sam3",
                        "model_id": args.model_id,
                    }
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1

            if (start // args.batch_size) % 10 == 0:
                print(f"Processed {min(start + args.batch_size, len(img_paths))}/{len(img_paths)} | crops: {n_written}")

    print("Saved manifest:", manifest_path)
    print("Total crops saved:", n_written)
    print("Out dir:", out_dir)


if __name__ == "__main__":
    main()