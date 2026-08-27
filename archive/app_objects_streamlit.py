import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel

st.set_page_config(page_title="Object-CBIR (DINO)", layout="wide")
DEFAULT_OBJ_DUMP = "dump_obj_dino.pk1"


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


@st.cache_resource
def load_dino():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()
    return device, proc, model


def embed(img: Image.Image, device, proc, model) -> np.ndarray:
    inputs = proc(images=img.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        out = model(pixel_values)
        e = out.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return normalize(e)


@st.cache_data
def load_object_dump(obj_dump: str):
    with open(obj_dump, "rb") as f:
        payload = pickle.load(f)

    emb_dict = payload["embeddings"]  # crop_path -> vec
    meta = payload["meta"]           # crop_path -> record

    paths = list(emb_dict.keys())
    embs = np.stack([emb_dict[p] for p in paths], axis=0).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return paths, embs, meta


st.title("CBIR hiérarchique (objets) — DINOv2")

obj_dump = st.text_input("Dump objets (.pk1)", value=DEFAULT_OBJ_DUMP)
topk = st.slider("Top-K objets", 1, 50, 10)
uploaded = st.file_uploader("Upload un objet/crop (requête)", type=["jpg", "jpeg", "png"])

if not os.path.exists(obj_dump):
    st.error(f"Dump introuvable: {obj_dump}")
    st.stop()

paths, embs, meta = load_object_dump(obj_dump)
device, proc, model = load_dino()

if uploaded is None:
    st.info("Upload une image d'objet (crop) pour lancer la recherche.")
    st.stop()

q_img = Image.open(uploaded)
st.subheader("Objet requête")
st.image(q_img, use_container_width=True)

q = embed(q_img, device, proc, model)
sims = embs @ q
idx = np.argsort(-sims)[:topk]
results = [(paths[i], float(sims[i])) for i in idx]

st.subheader("Objets similaires")
cols = st.columns(5)
for j, (p, s) in enumerate(results):
    c = cols[j % 5]
    rec = meta.get(p, {})
    try:
        img = Image.open(p).convert("RGB")
        cap = f"{j+1}. score={s:.3f}\n{Path(p).name}\n{rec.get('label_name','')}"
        c.image(img, caption=cap, use_container_width=True)
    except Exception:
        c.write(f"{j+1}. score={s:.3f}")
        c.write(Path(p).name)
    parent = rec.get("parent_path", None)
    if parent:
        c.caption(f"Parent: {Path(parent).name}")