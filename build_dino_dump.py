#!/usr/bin/env python3
import os
import pickle
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from dotenv import load_dotenv

EXTS = ("*.jpg", "*.jpeg", "*.png")


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def collect_images(root: Path):
    paths = []
    for ext in EXTS:
        paths += glob(str(root / "**" / ext), recursive=True)
    return sorted(paths)

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    paths, pixels = zip(*batch)
    return list(paths), torch.stack(pixels, dim=0)

class DinoImageDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return None
        inputs = self.processor(images=img, return_tensors="pt")
        return path, inputs["pixel_values"].squeeze(0)


def main():
    load_dotenv(script_dir() / "test.env")

    ref_folder = Path(os.getenv("REF_IMAGES_FOLDER", "")).expanduser()
    dump_path = os.getenv("DINO_DUMP_PATH", "dump_dino.pk1")
    dump_path = script_dir() / dump_path

    batch_size = int(os.getenv("DINO_BATCH_SIZE", "32"))
    num_workers = int(os.getenv("DINO_NUM_WORKERS", "0"))  # Windows-safe

    if not ref_folder.exists():
        raise FileNotFoundError(f"REF_IMAGES_FOLDER not found: {ref_folder}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()

    image_paths = collect_images(ref_folder)
    print(f"Found {len(image_paths)} images")

    ds = DinoImageDataset(image_paths, processor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_skip_none)

    embeddings = {}
    with torch.no_grad():
        for i, batch in enumerate(dl, start=1):
            if batch is None:
                continue
            paths, pixel_values = batch
            pixel_values = pixel_values.to(device)
            outputs = model(pixel_values)
            embs = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().astype(np.float32)
            for p, e in zip(paths, embs):
                key = os.path.normpath(str(p))
                embeddings[key] = e
            if i % 10 == 0:
                print(f"Batch {i}/{len(dl)}")

    with open(dump_path, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Saved DINO dump: {dump_path} ({len(embeddings)} items)")


if __name__ == "__main__":
    main()