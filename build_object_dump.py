#!/usr/bin/env python3
import json
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from dotenv import load_dotenv

def script_dir() -> Path:
    return Path(__file__).resolve().parent

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    paths, pixels = zip(*batch)
    return list(paths), torch.stack(pixels, dim=0)

class CropDataset(Dataset):
    def __init__(self, crop_paths, processor):
        self.crop_paths = crop_paths
        self.processor = processor

    def __len__(self):
        return len(self.crop_paths)

    def __getitem__(self, idx):
        p = self.crop_paths[idx]
        try:
            img = Image.open(p)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                img = img.convert("RGBA")
                # fond neutre (évite le blanc, et évite un noir trop dur)
                bg = Image.new("RGBA", img.size, (127, 127, 127, 255))
                img = Image.alpha_composite(bg, img).convert("RGB")
            else:
                img = img.convert("RGB")
        except Exception:
            return None
        inputs = self.processor(images=img, return_tensors="pt")
        return p, inputs["pixel_values"].squeeze(0)

def main():
    load_dotenv(script_dir() / "test.env")

    manifest_path = Path(os.getenv("OBJ_MANIFEST_PATH", "objects_manifest.jsonl"))
    if not manifest_path.is_absolute():
        manifest_path = script_dir() / manifest_path

    out_dump = Path(os.getenv("OBJ_DINO_DUMP_PATH", "dump_obj_dino.pk1"))
    if not out_dump.is_absolute():
        out_dump = script_dir() / out_dump

    batch_size = int(os.getenv("OBJ_DINO_BATCH_SIZE", "64"))
    num_workers = int(os.getenv("OBJ_DINO_NUM_WORKERS", "0"))

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records = []
    crop_paths = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cp = r["crop_path"]
            if os.path.exists(cp):
                records.append(r)
                crop_paths.append(cp)

    print(f"Found {len(crop_paths)} object crops")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()

    ds = CropDataset(crop_paths, processor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, collate_fn=collate_skip_none)

    emb = {}
    with torch.no_grad():
        for i, batch in enumerate(dl, start=1):
            if batch is None:
                continue
            paths, pixel_values = batch
            pixel_values = pixel_values.to(device)
            out = model(pixel_values)
            e = out.last_hidden_state.mean(dim=1).detach().cpu().numpy().astype(np.float32)
            for p, vec in zip(paths, e):
                emb[os.path.normpath(str(p))] = vec
            if i % 10 == 0:
                print(f"Batch {i}/{len(dl)}")

    meta = {os.path.normpath(r["crop_path"]): r for r in records}
    payload = {"embeddings": emb, "meta": meta}
    with open(out_dump, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved object dump: {out_dump}")

if __name__ == "__main__":
    main()