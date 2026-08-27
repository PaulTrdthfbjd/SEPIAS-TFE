#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def load_dump(dump_path: str):
    with open(dump_path, "rb") as f:
        ref = pickle.load(f)  # dict path->emb
    paths = list(ref.keys())
    embs = np.stack([ref[p] for p in paths], axis=0).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return paths, embs


def compute_dino_embedding(img_path: str, device: str):
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()

    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        out = model(pixel_values)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)

    return normalize(emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=str, required=True, help="DINO dump (.pk1)")
    ap.add_argument("--query", type=str, required=True, help="Path to query image")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--exclude_self", action="store_true", help="Exclude query if it exists in dump")
    args = ap.parse_args()

    if not os.path.exists(args.dump):
        raise FileNotFoundError(args.dump)
    if not os.path.exists(args.query):
        raise FileNotFoundError(args.query)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = compute_dino_embedding(args.query, device)

    ref_paths, ref_embs = load_dump(args.dump)
    sims = ref_embs @ q

    if args.exclude_self:
        try:
            idx = ref_paths.index(str(Path(args.query)))
            sims[idx] = -np.inf
        except ValueError:
            pass

    top_idx = np.argsort(-sims)[: args.topk]
    for rank, i in enumerate(top_idx, start=1):
        print(f"{rank:02d} sim={sims[i]:.4f}  {ref_paths[i]}")


if __name__ == "__main__":
    main()