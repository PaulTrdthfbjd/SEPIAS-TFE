#!/usr/bin/env python3


import os
from glob import glob
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from dotenv import load_dotenv
import pickle

# ---------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------
load_dotenv(Path(__file__).with_name("test.env"))

REF_IMAGES_FOLDER = Path(os.getenv("REF_IMAGES_FOLDER"))
QUERY_IMAGE_PATH = Path(os.getenv("QUERY_IMAGE_PATH"))
NB_IMG_OUTPUT =  int(os.getenv("NB_IMG_OUTPUT"))


# ---------------------------------------------------------------------
# Device setup and load model
# ---------------------------------------------------------------------

# Use gpu if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} with {torch.cuda.device_count()} GPUs")

# Use pretrained DINOv2
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

# Multi-GPU support
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs with DataParallel")
    model = nn.DataParallel(model)

model = model.to(device)
model.eval()


# ---------------------------------------------------------------------
# Classes and functions
# ---------------------------------------------------------------------

# Dataset class adapted for DINOv2
class DinoImageDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        # 'pixel_values' has shape (1, 3, H, W) → remove batch dimension
        return str(path), inputs['pixel_values'].squeeze(0)

    def __len__(self):
        return len(self.image_paths)


# Compute the similarity between two embeddings
def pairwise_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Given a list of paths, build dataloader and compute and return a list of embeddings
def get_embeddings_from_paths(image_paths, model, device, processor, batch_size=32):
    dataset = DinoImageDataset(image_paths, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    embeddings = {}
    model.eval()

    with torch.no_grad():
        for i, (paths, pixel_values) in enumerate(dataloader):
            pixel_values = pixel_values.to(device)
            outputs = model(pixel_values)

            # DINOv2’s main representation → last_hidden_state
            # shape: (B, n_patches, hidden_dim)
            embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            for path, emb in zip(paths, embs):
                embeddings[path] = emb

            print(f"Processing batch {i+1}/{len(dataloader)}", end='\r')

    print("\nDone.")
    return embeddings


def show_images_grid(top_images, ncols=5):
    """
    Display images in a single window with their filename and rank/score.
    top_images: list of (path, similarity) tuples
    ncols: number of images per row
    """
    n = len(top_images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))

    # Flatten axes for easy iteration
    axes = axes.flatten() if n > 1 else [axes]

    for i, (path, similarity) in enumerate(top_images):
        image = Image.open(path)
        ax = axes[i]
        ax.imshow(np.array(image))
        ax.axis("off")

        # filename only (no full path)
        filename = os.path.basename(path)
        ax.set_title(f"{i}. {filename}\n(sim={similarity:.3f})",
                     fontsize=9, pad=4)

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show(block=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Get the files from the env file
    ref_images_folder = REF_IMAGES_FOLDER
    query_image_path = QUERY_IMAGE_PATH
    nb_img_output = NB_IMG_OUTPUT
    ref_dump_path = f"./dump_{os.path.basename(ref_images_folder)}.pkl"

    # Load ref images and compute embeddings
    # If pre-computed embeddings exist, load them
    if os.path.exists(ref_dump_path):
        print(f"Existing reference found, loading {ref_dump_path}")
        with open(ref_dump_path, "rb") as dump:
            reference = pickle.load(dump)
    # If they don't exist, compute them
    else:
        print("Existing reference not found, computing now")
        ref_images_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            ref_images_paths += glob(os.path.join(ref_images_folder, "**", ext), recursive=True)
        print(f"Found {len(ref_images_paths)} images")

        # Compute all embeddings
        reference = get_embeddings_from_paths(ref_images_paths, model, device, processor)

        # Save embeddings
        with open(ref_dump_path, "wb") as dump:
            print("Save reference as local dump")
            pickle.dump(reference, dump)

    # -----------------------------------------------------------------
    # Standard CBIR (no interaction): 1 query -> top-K nearest neighbors
    # -----------------------------------------------------------------

    # Compute query embedding
    query_path_str = str(query_image_path)
    query_emb = next(iter(get_embeddings_from_paths([query_path_str], model, device, processor).values()))

    # Build matrix of reference embeddings (vectorized similarity)
    ref_paths = list(reference.keys())
    ref_embs = np.stack([reference[p] for p in ref_paths], axis=0)  # shape (N, D)

    # Optional: remove the query itself if it belongs to the reference set
    # (useful when the query image is inside REF_IMAGES_FOLDER)
    if query_path_str in reference:
        # We'll filter it out from ranking
        mask = np.array([p != query_path_str for p in ref_paths], dtype=bool)
        ref_paths = [p for p, m in zip(ref_paths, mask) if m]
        ref_embs = ref_embs[mask]

    # Normalize for cosine similarity: sim = dot(normalized vectors)
    ref_embs = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-12)
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-12)

    sims = ref_embs @ query_emb  # shape (N,)
    topk_idx = np.argsort(-sims)[:nb_img_output]

    top_images = [(ref_paths[i], float(sims[i])) for i in topk_idx]

    print("Top results:")
    for rank, (p, s) in enumerate(top_images, start=1):
        print(f"{rank:02d}  sim={s:.4f}  {p}")

    # Display results
    show_images_grid(top_images)

    # Save ranking to CSV (handy for reporting)
    out_csv = "cbir_results.csv"
    pd.DataFrame(
        {"rank": np.arange(1, len(top_images)+1),
         "path": [p for p, _ in top_images],
         "similarity": [s for _, s in top_images]}
    ).to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")
