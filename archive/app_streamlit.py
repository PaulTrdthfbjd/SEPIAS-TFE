import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

st.set_page_config(page_title="CBIR (DINOv2)", layout="wide")
DEFAULT_DUMP = "dump_dino.pk1"


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()
    return device, processor, model


def compute_embedding(img: Image.Image, device, processor, model) -> np.ndarray:
    img = img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        outputs = model(pixel_values)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return emb


@st.cache_data
def load_reference(dump_path: str):
    with open(dump_path, "rb") as f:
        ref = pickle.load(f)
    paths = list(ref.keys())
    embs = np.stack([ref[p] for p in paths], axis=0).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return paths, embs


st.title("CBIR (DINOv2)")

dump_path = st.text_input("Dump embeddings (.pk1)", value=DEFAULT_DUMP)
k = st.slider("Top-K", 1, 50, 10)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    uploaded = st.file_uploader("Upload image requête", type=["jpg", "jpeg", "png"])

with col2:
    if not os.path.exists(dump_path):
        st.error(f"Dump introuvable: {dump_path}")
        st.stop()

    ref_paths, ref_embs = load_reference(dump_path)
    device, processor, model = load_model()

    if uploaded is None:
        st.info("Upload une image pour lancer la recherche.")
        st.stop()

    query_img = Image.open(uploaded)
    st.subheader("Image requête")
    st.image(query_img, use_container_width=True)

    q = normalize(compute_embedding(query_img, device, processor, model))
    sims = ref_embs @ q
    top_idx = np.argsort(-sims)[:k]
    results = [(ref_paths[i], float(sims[i])) for i in top_idx]

    st.subheader("Résultats")
    cols = st.columns(5)
    for idx, (path, sim) in enumerate(results):
        c = cols[idx % 5]
        try:
            img = Image.open(path)
            c.image(img, caption=f"{idx+1}. sim={sim:.3f}\n{Path(path).name}", use_container_width=True)
        except Exception:
            c.write(f"{idx+1}. sim={sim:.3f}")
            c.write(Path(path).name)