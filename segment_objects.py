#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.transforms.functional import to_tensor
from dotenv import load_dotenv
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)

EXTS = ("*.jpg", "*.jpeg", "*.png")

# COCO labels for torchvision (index aligns with model output label ids)
COCO_INSTANCE_CATEGORIES = [
    "__background__",
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","N/A","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","N/A","backpack",
    "umbrella","N/A","N/A","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","N/A",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","N/A","dining table",
    "N/A","N/A","toilet","N/A","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","N/A","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_images", type=int, default=500, help="Limit images for a first run")
    ap.add_argument("--score_thr", type=float, default=0.7)
    ap.add_argument("--max_objs_per_image", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="objects")
    args = ap.parse_args()

    load_dotenv(script_dir() / "test.env")
    ref_folder = Path(os.getenv("REF_IMAGES_FOLDER", "")).expanduser()
    manifest_path = os.getenv("OBJ_MANIFEST_PATH", "objects_manifest.jsonl")
    manifest_path = script_dir() / manifest_path

    out_dir = script_dir() / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ref_folder.exists():
        raise FileNotFoundError(f"REF_IMAGES_FOLDER not found: {ref_folder}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Mask R-CNN pretrained
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()

    img_paths = collect_images(ref_folder)
    if args.max_images > 0:
        img_paths = img_paths[: min(args.max_images, len(img_paths))]
    print(f"Processing {len(img_paths)} images")

    n_written = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for idx, p in enumerate(img_paths, start=1):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue

            x = preprocess(img).to(device)

            with torch.no_grad():
                pred = model([x])[0]

            scores = pred["scores"].detach().cpu().numpy()
            labels = pred["labels"].detach().cpu().numpy()
            masks = pred["masks"].detach().cpu().numpy()  # (N,1,H,W)

            # filter and take top objects
            keep = np.where(scores >= args.score_thr)[0]
            keep = keep[: args.max_objs_per_image]

            if len(keep) == 0:
                continue

            img_np = np.array(img)
            H, W = img_np.shape[:2]

            for j in keep:
                mask = masks[j, 0] > 0.5
                if mask.sum() < 100:  # too small
                    continue

                ys, xs = np.where(mask)
                y1, y2 = int(ys.min()), int(ys.max())
                x1, x2 = int(xs.min()), int(xs.max())

                # pad a bit
                pad = 5
                x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)

                crop = img_np[y1:y2+1, x1:x2+1]

                lab_id = int(labels[j])
                lab_name = COCO_INSTANCE_CATEGORIES[lab_id] if lab_id < len(COCO_INSTANCE_CATEGORIES) else str(lab_id)
                lab_name_safe = str(lab_name).replace("/", "_").replace("\\", "_").replace(" ", "_")

                obj_id = f"{idx:06d}_{j:02d}"
                crop_path = out_dir / f"{obj_id}_{lab_name_safe}.png"
                Image.fromarray(crop, mode="RGB").save(crop_path)

                record = {
                    "obj_id": obj_id,
                    "label_id": lab_id,
                    "label_name": lab_name,
                    "score": float(scores[j]),
                    "parent_path": str(p),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "crop_path": str(crop_path)
                }
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(img_paths)} images | objects saved: {n_written}")

    print(f"Saved manifest: {manifest_path}")
    print(f"Total objects saved: {n_written}")


if __name__ == "__main__":
    main()