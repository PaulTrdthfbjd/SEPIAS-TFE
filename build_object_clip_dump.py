#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def normpath(p: str) -> str:
    return os.path.normpath(str(p))


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    paths, pixels = zip(*batch)
    return list(paths), torch.stack(pixels, dim=0)


def open_rgb_safely(path: str) -> Image.Image | None:
    """
    Ouvre une image en RGB. Si elle contient de l'alpha (RGBA), on la "flatten"
    sur un fond gris moyen (évite une transparence/alpha buggée).
    """
    try:
        img = Image.open(path)
    except Exception:
        return None

    try:
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (127, 127, 127, 255))
            img = Image.alpha_composite(bg, img).convert("RGB")
        else:
            img = img.convert("RGB")
        return img
    except Exception:
        return None


class ClipCropDataset(Dataset):
    def __init__(self, crop_paths, processor):
        self.crop_paths = crop_paths
        self.processor = processor

    def __len__(self):
        return len(self.crop_paths)

    def __getitem__(self, idx):
        p = self.crop_paths[idx]
        img = open_rgb_safely(p)
        if img is None:
            return None
        inputs = self.processor(images=img, return_tensors="pt")
        return p, inputs["pixel_values"].squeeze(0)


def read_manifest_jsonl(manifest_path: Path):
    records = []
    crop_paths = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cp = r.get("crop_path", None)
            if not cp:
                continue
            cp = normpath(cp)
            if os.path.exists(cp):
                # normalise aussi parent_path dans meta (utile côté streamlit)
                if "parent_path" in r and r["parent_path"]:
                    r["parent_path"] = normpath(r["parent_path"])
                r["crop_path"] = cp
                records.append(r)
                crop_paths.append(cp)
    return records, crop_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=None, help="Override manifest jsonl (sinon OBJ_MANIFEST_PATH)")
    ap.add_argument("--out_dump", type=str, default=None, help="Override out dump path (sinon OBJ_CLIP_DUMP_PATH)")
    ap.add_argument("--model_id", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch size (sinon OBJ_CLIP_BATCH_SIZE)")
    ap.add_argument("--num_workers", type=int, default=None, help="Override num_workers (sinon OBJ_CLIP_NUM_WORKERS)")
    args = ap.parse_args()

    load_dotenv(script_dir() / "test.env")

    # --- Paths / params from env fallback ---
    manifest_path = Path(args.manifest or os.getenv("OBJ_MANIFEST_PATH", "objects_manifest.jsonl"))
    if not manifest_path.is_absolute():
        manifest_path = script_dir() / manifest_path

    out_dump = Path(args.out_dump or os.getenv("OBJ_CLIP_DUMP_PATH", "dump_obj_clip.pk1"))
    if not out_dump.is_absolute():
        out_dump = script_dir() / out_dump

    batch_size = args.batch_size if args.batch_size is not None else int(os.getenv("OBJ_CLIP_BATCH_SIZE", "64"))
    num_workers = args.num_workers if args.num_workers is not None else int(os.getenv("OBJ_CLIP_NUM_WORKERS", "0"))

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records, crop_paths = read_manifest_jsonl(manifest_path)
    print(f"Found {len(crop_paths)} crop files from manifest: {manifest_path}")

    if len(crop_paths) == 0:
        raise RuntimeError("0 crops trouvés. Vérifie crop_path dans le manifest + existence des fichiers.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = CLIPProcessor.from_pretrained(args.model_id)
    model = CLIPModel.from_pretrained(args.model_id).to(device)
    model.eval()

    ds = ClipCropDataset(crop_paths, processor)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_skip_none,
    )

    embeddings = {}
    with torch.no_grad():
        for i, batch in enumerate(dl, start=1):
            if batch is None:
                continue
            paths, pixel_values = batch
            pixel_values = pixel_values.to(device)

            embs = model.get_image_features(pixel_values=pixel_values)  # (B, D)
            embs = embs.detach().cpu().numpy().astype(np.float32)

            for p, e in zip(paths, embs):
                embeddings[normpath(p)] = e

            if i % 10 == 0 or i == len(dl):
                print(f"Batch {i}/{len(dl)}")

    meta = {normpath(r["crop_path"]): r for r in records}
    payload = {"embeddings": embeddings, "meta": meta}

    out_dump.parent.mkdir(parents=True, exist_ok=True)
    with open(out_dump, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved CLIP object dump: {out_dump} ({len(embeddings)} items)")


if __name__ == "__main__":
    main()