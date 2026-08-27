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
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import pickle

# ---------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------
load_dotenv(Path("./test.env"))

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


# Compute the similarity from 1 to n embeddings (defined as the average of pairwise similarities)
def onetomany_similarity(embedding1, embedding_list):
    all_pairwise_sim = [pairwise_similarity(embedding1, emb) for emb in embedding_list]
    return sum(all_pairwise_sim)/len(all_pairwise_sim)


# Compute the similarity score using both positive and negative query embeddings
def posneg_similarity(embedding1, pos_embeddings, neg_embeddings):
    pos_sim = [pairwise_similarity(embedding1, emb) for emb in pos_embeddings]
    neg_sim = [pairwise_similarity(embedding1, emb) for emb in neg_embeddings]
    if len(pos_sim) == 0 and len(neg_sim) == 0:
        return 0.0
    elif len(pos_sim) == 0:
        return -np.mean(neg_sim)
    elif len(neg_sim) == 0:
        return np.mean(pos_sim)
    else:
        return np.mean(pos_sim) - np.mean(neg_sim)



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
        ref_images_paths = glob(os.path.join(ref_images_folder, '**/*.jpg'), recursive=True)
        print(f"Found {len(ref_images_paths)} images")

        # Compute all embeddings
        reference = get_embeddings_from_paths(ref_images_paths, model, device, processor)

        # Save embeddings
        with open(ref_dump_path, "wb") as dump:
            print("Save reference as local dump")
            pickle.dump(reference, dump)

    # Iterative search
    # Start with one query
    pos_query_list = [query_image_path]
    pos_query_embeddings = [next(iter(get_embeddings_from_paths(pos_query_list, model, device, processor).values()))]
    neg_query_list = []
    neg_query_embeddings = []

    # Iterate by selecting positive queries
    while True:
        print("-"*80)
        # Compute similarities w.r.t. the multi-image query
        similarities = [(path, posneg_similarity(ref_emb, pos_query_embeddings, neg_query_embeddings)) for path, ref_emb in reference.items()]
        # Select the most similar images
        top_images = sorted(similarities, key=lambda x: x[1], reverse=True)[:nb_img_output]
        # Display them
        show_images_grid(top_images)
        # Ask user for images to add to query
        new_pos_query = list(map(int, input("Select pictures for positive query (space-separated numbers): ").split()))
        new_neg_query = list(map(int, input("Select pictures for negative query (space-separated numbers): ").split()))
        # Stop the search if no more query images are provided
        if not new_pos_query and not new_neg_query:
            break
        # Append new images to the query and retrieve their embeddings
        print("Selected for positive query")
        for i in new_pos_query:
            p = top_images[i][0] # Path of the image to add to the query
            print(p)
            pos_query_list.append(p)
            pos_query_embeddings.append(reference.pop(p)) # Retrieve embedding and removes the element from dictionary 'reference'
        print("Selected for negative query")
        for i in new_neg_query:
            p = top_images[i][0] # Path of the image to add to the query
            print(p)
            neg_query_list.append(p)
            neg_query_embeddings.append(reference.pop(p))

    print("Final output")
    for path in pos_query_list:
        print(path)
    for path, _ in top_images:
        print(path)
