import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel

st.set_page_config(page_title="Adaptive CBIR (DINO+CLIP)", layout="wide")


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


@st.cache_resource
def load_dino():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()
    return device, proc, model


@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return device, proc, model


@st.cache_data
def load_dump_matrix(dump_path: str):
    with open(dump_path, "rb") as f:
        ref = pickle.load(f)
    paths = list(ref.keys())
    embs = np.stack([ref[p] for p in paths], axis=0).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return paths, embs


def dino_embed(img: Image.Image, device, proc, model) -> np.ndarray:
    inputs = proc(images=img.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        out = model(pixel_values)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return normalize(emb)


def clip_img_embed(img: Image.Image, device, proc, model) -> np.ndarray:
    inputs = proc(images=img.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        emb = model.get_image_features(pixel_values=pixel_values).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return emb


def clip_text_embed(text: str, device, proc, model) -> np.ndarray:
    inputs = proc(text=[text], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        emb = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return emb


st.title("Moteur CBIR adaptable (DINOv2 + CLIP)")

colA, colB = st.columns([1, 2], gap="large")

with colA:
    dino_dump = st.text_input("Dump DINO", value="dump_dino.pk1")
    clip_dump = st.text_input("Dump CLIP", value="dump_clip.pk1")
    topk = st.slider("Top-K", 1, 50, 10)

    alpha = st.slider("α (poids DINO) : style/structure ↔ contenu", 0.0, 1.0, 0.7, 0.05)
    text_prompt = st.text_input("Texte (optionnel) pour CLIP", value="")
    text_weight = st.slider("Poids texte (CLIP)", 0.0, 2.0, 0.7, 0.05)

    uploaded = st.file_uploader("Upload image requête", type=["jpg", "jpeg", "png"])

with colB:
    if not os.path.exists(dino_dump):
        st.error(f"Dump DINO introuvable: {dino_dump}")
        st.stop()

    dino_paths, dino_embs = load_dump_matrix(dino_dump)

    use_clip = os.path.exists(clip_dump)
    if use_clip:
        clip_paths, clip_embs = load_dump_matrix(clip_dump)
        if set(clip_paths) != set(dino_paths):
            st.warning("DINO/CLIP dumps ne couvrent pas les mêmes images. Mode DINO-only.")
            use_clip = False
        else:
            idx_map = {p: i for i, p in enumerate(clip_paths)}
            clip_embs = np.stack([clip_embs[idx_map[p]] for p in dino_paths], axis=0)

    d_dev, d_proc, d_model = load_dino()
    if use_clip:
        c_dev, c_proc, c_model = load_clip()

    if uploaded is None:
        st.info("Upload une image pour lancer la recherche.")
        st.stop()

    q_img = Image.open(uploaded)
    st.subheader("Image requête")
    st.image(q_img, use_container_width=True)

    q_dino = dino_embed(q_img, d_dev, d_proc, d_model)
    sims = dino_embs @ q_dino

    if use_clip:
        q_ci = clip_img_embed(q_img, c_dev, c_proc, c_model)
        q_c = q_ci
        if text_prompt.strip():
            q_ct = clip_text_embed(text_prompt.strip(), c_dev, c_proc, c_model)
            q_c = q_ci + text_weight * q_ct
        q_c = normalize(q_c)

        sims = alpha * (dino_embs @ q_dino) + (1.0 - alpha) * (clip_embs @ q_c)

    top_idx = np.argsort(-sims)[:topk]
    results = [(dino_paths[i], float(sims[i])) for i in top_idx]

    st.subheader("Résultats")
    cols = st.columns(5)
    for j, (p, s) in enumerate(results):
        c = cols[j % 5]
        try:
            img = Image.open(p)
            c.image(img, caption=f"{j+1}. score={s:.3f}\n{Path(p).name}", use_container_width=True)
        except Exception:
            c.write(f"{j+1}. score={s:.3f}")
            c.write(Path(p).name)