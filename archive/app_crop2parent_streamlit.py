import os
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

import torch
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel

st.set_page_config(page_title="Crop → Parent CBIR (DINO/CLIP/Fusion)", layout="wide")

# -----------------------------
# Canonical paths (Windows-safe)
# -----------------------------
def canon(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p)))

def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

# -----------------------------
# Dump loaders
# -----------------------------
@st.cache_data
def load_vec_dump(dump_path: str):
    """Dump = dict[path -> vec]"""
    with open(dump_path, "rb") as f:
        ref = pickle.load(f)
    ref2 = {canon(k): v for k, v in ref.items()}
    paths = list(ref2.keys())
    embs = np.stack([ref2[p] for p in paths], axis=0).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return paths, embs

@st.cache_data
def load_object_dump(obj_dump_path: str):
    """Object dump = {'embeddings': {crop_path->vec}, 'meta': {crop_path->record}}"""
    with open(obj_dump_path, "rb") as f:
        payload = pickle.load(f)

    emb_dict = payload["embeddings"]
    meta = payload["meta"]

    # Keep a consistent order
    raw_paths = list(emb_dict.keys())
    crop_paths = [canon(p) for p in raw_paths]
    embs = np.stack([emb_dict[p] for p in raw_paths], axis=0).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    # Normalize meta keys and inner paths
    meta_norm = {}
    for k, v in meta.items():
        kk = canon(k)
        vv = dict(v)
        if "parent_path" in vv:
            vv["parent_path"] = canon(vv["parent_path"])
        if "crop_path" in vv:
            vv["crop_path"] = canon(vv["crop_path"])
        meta_norm[kk] = vv

    return crop_paths, embs, meta_norm

# -----------------------------
# Models
# -----------------------------
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

def dino_embed(img: Image.Image, device, proc, model) -> np.ndarray:
    img = img.convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        out = model(pixel_values)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return normalize(emb)

def clip_embed(img: Image.Image, device, proc, model) -> np.ndarray:
    img = img.convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        emb = model.get_image_features(pixel_values=pixel_values).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return normalize(emb)

# -----------------------------
# Crop → Parent utilities
# -----------------------------
def draw_bbox(parent_img: Image.Image, bbox_xyxy, width=6):
    img = parent_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox_xyxy
    for w in range(width):
        draw.rectangle([x1-w, y1-w, x2+w, y2+w], outline=(255, 0, 0))
    return img

def rank_parents_from_crop_scores(crop_paths, crop_scores, meta, agg="max", top_per_parent=3):
    bucket = defaultdict(list)  # parent_path -> list[(score, crop_path, bbox)]
    for p, s in zip(crop_paths, crop_scores):
        rec = meta.get(canon(p))
        if rec is None:
            continue
        parent = rec.get("parent_path", None)
        bbox = rec.get("bbox_xyxy", None)
        if parent is None:
            continue
        bucket[parent].append((float(s), canon(p), bbox))

    parent_rows = []
    for parent, lst in bucket.items():
        lst.sort(key=lambda t: t[0], reverse=True)
        top_lst = lst[:top_per_parent]

        if agg == "max":
            parent_score = top_lst[0][0]
        elif agg == "mean":
            parent_score = float(np.mean([t[0] for t in top_lst]))
        else:
            raise ValueError("agg must be 'max' or 'mean'")

        parent_rows.append({
            "parent_path": parent,
            "parent_score": parent_score,
            "matches": top_lst,
        })

    parent_rows.sort(key=lambda r: r["parent_score"], reverse=True)
    return parent_rows

# -----------------------------
# UI
# -----------------------------
st.title("Crop → Parent retrieval (DINO / CLIP / Fusion)")

colL, colR = st.columns([1, 2], gap="large")

with colL:
    obj_dump = st.text_input("Object dump (DINO) (.pk1)", value="dump_obj_segnet_scenes.pk1")

    mode = st.radio("Mode scoring", ["dino", "clip", "fusion"], index=2, horizontal=True)
    alpha = st.slider("alpha (poids DINO en fusion)", 0.0, 1.0, 0.7, 0.05)

    crop_clip_dump = st.text_input("Dump CLIP crops (.pk1 dict[path->vec])", value="dump_clip_obj_segnet_scenes.pk1")

    baseline_dino_dump = st.text_input("Dump baseline DINO (full images)", value="dump_dino_scenes.pk1")
    baseline_clip_dump = st.text_input("Dump baseline CLIP (full images)", value="dump_clip_scenes.pk1")

    topk_baseline = st.slider("Top-K baseline", 1, 30, 10)

    topk_parents = st.slider("Top-K parents", 1, 30, 10)
    topn_crops = st.slider("Top-N crops (candidats)", 10, 5000, 200, step=10)
    agg = st.radio("Agrégation par parent", ["max", "mean"], index=0, horizontal=True)
    top_per_parent = st.slider("Nb crops affichés / parent", 1, 5, 1)
    show_bbox = st.checkbox("Afficher bbox sur l'image parent", value=True)

    uploaded = st.file_uploader("Upload une image crop (requête)", type=["jpg", "jpeg", "png"])

with colR:
    if not os.path.exists(obj_dump):
        st.error(f"Object dump introuvable: {obj_dump}")
        st.stop()

    # Load object dump (DINO crops + meta)
    crop_paths_dino, crop_embs_dino, meta = load_object_dump(obj_dump)

    # Load models
    d_dev, d_proc, d_model = load_dino()
    if mode in ("clip", "fusion"):
        c_dev, c_proc, c_model = load_clip()

    if uploaded is None:
        st.info("Upload un crop pour lancer la recherche.")
        st.stop()

    # Read query crop (handle transparency)
    q_img = Image.open(uploaded)
    if q_img.mode in ("RGBA", "LA") or (q_img.mode == "P" and "transparency" in q_img.info):
        q_img = q_img.convert("RGBA")
        bg = Image.new("RGBA", q_img.size, (127, 127, 127, 255))
        q_img = Image.alpha_composite(bg, q_img).convert("RGB")
    else:
        q_img = q_img.convert("RGB")

    st.subheader("Crop requête")
    st.image(q_img, use_container_width=True)

    # Query embeddings
    q_dino = dino_embed(q_img, d_dev, d_proc, d_model)
    q_clip = None
    if mode in ("clip", "fusion"):
        q_clip = clip_embed(q_img, c_dev, c_proc, c_model)

    # -----------------------------
    # Prepare CROP embeddings per mode (and align if needed)
    # -----------------------------
    crop_paths = crop_paths_dino
    crop_scores = None

    if mode == "dino":
        sims_crops = crop_embs_dino @ q_dino
        crop_scores = sims_crops

    else:
        # Need CLIP crop dump
        if not os.path.exists(crop_clip_dump):
            st.error(f"Dump CLIP crops introuvable: {crop_clip_dump} (mode {mode})")
            st.stop()

        clip_paths, clip_embs = load_vec_dump(crop_clip_dump)
        clip_idx = {p: i for i, p in enumerate(clip_paths)}

        # align on DINO crop list (to keep meta consistent)
        keep = [p for p in crop_paths_dino if p in clip_idx]
        if len(keep) == 0:
            st.error("Aucun crop commun entre object dump (DINO) et dump CLIP crops. Vérifie les chemins/normalisation.")
            st.stop()

        dino_idx = [crop_paths_dino.index(p) for p in keep]
        clip_idxs = [clip_idx[p] for p in keep]

        crop_paths = keep
        crop_embs_dino_al = crop_embs_dino[dino_idx]
        crop_embs_clip_al = clip_embs[clip_idxs]

        sims_d = crop_embs_dino_al @ q_dino
        sims_c = crop_embs_clip_al @ q_clip

        if mode == "clip":
            crop_scores = sims_c
        else:  # fusion
            crop_scores = alpha * sims_d + (1.0 - alpha) * sims_c

    # -----------------------------
    # BASELINE full-image retrieval (same query crop embedding)
    # -----------------------------
    baseline_results = []
    if mode == "dino":
        if os.path.exists(baseline_dino_dump):
            full_paths, full_embs = load_vec_dump(baseline_dino_dump)
            sims_full = full_embs @ q_dino
            topb = np.argsort(-sims_full)[: int(topk_baseline)]
            baseline_results = [(full_paths[i], float(sims_full[i])) for i in topb]
        else:
            st.warning(f"Dump baseline DINO introuvable: {baseline_dino_dump}")

    elif mode == "clip":
        if os.path.exists(baseline_clip_dump):
            full_paths, full_embs = load_vec_dump(baseline_clip_dump)
            sims_full = full_embs @ q_clip
            topb = np.argsort(-sims_full)[: int(topk_baseline)]
            baseline_results = [(full_paths[i], float(sims_full[i])) for i in topb]
        else:
            st.warning(f"Dump baseline CLIP introuvable: {baseline_clip_dump}")

    else:  # fusion
        if not (os.path.exists(baseline_dino_dump) and os.path.exists(baseline_clip_dump)):
            st.warning("Fusion baseline: il faut les deux dumps (DINO+CLIP).")
        else:
            d_paths, d_embs = load_vec_dump(baseline_dino_dump)
            c_paths, c_embs = load_vec_dump(baseline_clip_dump)
            c_map = {p: i for i, p in enumerate(c_paths)}

            common = [p for p in d_paths if p in c_map]
            if len(common) == 0:
                st.warning("Fusion baseline: aucune image commune entre dumps DINO et CLIP.")
            else:
                d_map = {p: i for i, p in enumerate(d_paths)}
                d_idx = [d_map[p] for p in common]
                c_idx = [c_map[p] for p in common]
                sims_full = alpha * (d_embs[d_idx] @ q_dino) + (1.0 - alpha) * (c_embs[c_idx] @ q_clip)
                topb = np.argsort(-sims_full)[: int(topk_baseline)]
                baseline_results = [(common[i], float(sims_full[i])) for i in topb]

    # -----------------------------
    # Top crops → rank parents
    # -----------------------------
    topn = min(int(topn_crops), len(crop_paths))
    idx = np.argsort(-crop_scores)[:topn]
    top_crop_paths = [crop_paths[i] for i in idx]
    top_crop_scores = crop_scores[idx]

    tab1, tab2 = st.tabs(["Sans segmentation (baseline)", "Avec segmentation (crop→parent)"])

    with tab1:
        st.subheader("Résultats baseline (full images)")
        colsB = st.columns(5)
        for j, (p, s) in enumerate(baseline_results):
            c = colsB[j % 5]
            if os.path.exists(p):
                c.image(Image.open(p).convert("RGB"), caption=f"{j+1}. score={s:.3f}\n{Path(p).name}", use_container_width=True)
            else:
                c.write(f"{j+1}. {Path(p).name} (missing)")

    with tab2:
        parents_ranked = rank_parents_from_crop_scores(
            top_crop_paths, top_crop_scores, meta, agg=agg, top_per_parent=int(top_per_parent)
        )
        parents_ranked = parents_ranked[: int(topk_parents)]

        st.subheader("Résultats (images parent)")
        cols = st.columns(2)

        for j, row in enumerate(parents_ranked):
            parent_path = row["parent_path"]
            parent_score = row["parent_score"]
            matches = row["matches"]

            c = cols[j % 2]
            if not os.path.exists(parent_path):
                c.warning(f"Parent introuvable: {parent_path}")
                continue

            parent_img = Image.open(parent_path).convert("RGB")
            best_score, best_crop_path, best_bbox = matches[0]

            if show_bbox and best_bbox is not None:
                try:
                    parent_disp = draw_bbox(parent_img, best_bbox, width=6)
                except Exception:
                    parent_disp = parent_img
            else:
                parent_disp = parent_img

            c.image(
                parent_disp,
                caption=f"{j+1}. score_parent={parent_score:.3f}\n{Path(parent_path).name}",
                use_container_width=True
            )

            with c.expander("Détails du match"):
                c.write(f"Parent: {parent_path}")
                for k_idx, (s, cp, bb) in enumerate(matches, start=1):
                    c.write(f"- match#{k_idx} score={s:.3f} crop={Path(cp).name}")
                    if os.path.exists(cp):
                        c.image(Image.open(cp).convert('RGB'), caption=f"crop match#{k_idx}", use_container_width=True)