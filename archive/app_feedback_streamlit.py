import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel


st.set_page_config(page_title="Interactive CBIR Feedback", layout="wide")


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


@st.cache_resource
def load_dino():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()
    return device, processor, model


@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return device, processor, model


@st.cache_data
def load_dump_matrix(dump_path: str):
    with open(dump_path, "rb") as f:
        ref = pickle.load(f)

    paths = list(ref.keys())
    embs = np.stack([ref[p] for p in paths], axis=0).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return paths, embs


def dino_embed(img: Image.Image, device, processor, model) -> np.ndarray:
    img = img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        emb = emb.detach().cpu().numpy().astype(np.float32)

    return normalize(emb)


def clip_image_embed(img: Image.Image, device, processor, model) -> np.ndarray:
    img = img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        emb = model.get_image_features(pixel_values=pixel_values).squeeze(0)
        emb = emb.detach().cpu().numpy().astype(np.float32)

    return normalize(emb)


def clip_text_embed(text: str, device, processor, model) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        emb = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).squeeze(0)
        emb = emb.detach().cpu().numpy().astype(np.float32)

    return normalize(emb)


def align_clip_to_dino_paths(dino_paths, clip_paths, clip_embs):
    clip_map = {p: i for i, p in enumerate(clip_paths)}

    if set(dino_paths) != set(clip_paths):
        common = [p for p in dino_paths if p in clip_map]
        if len(common) == 0:
            raise RuntimeError("Aucune image commune entre les dumps DINO et CLIP.")

        dino_keep_idx = [i for i, p in enumerate(dino_paths) if p in clip_map]
        clip_keep_idx = [clip_map[p] for p in common]
        return common, dino_keep_idx, clip_embs[clip_keep_idx]

    clip_aligned = np.stack([clip_embs[clip_map[p]] for p in dino_paths], axis=0)
    return dino_paths, list(range(len(dino_paths))), clip_aligned


def compute_scores(
    mode: str,
    alpha: float,
    dino_embs: np.ndarray,
    clip_embs: np.ndarray | None,
    q_dino: np.ndarray | None,
    q_clip: np.ndarray | None,
):
    if mode == "DINO":
        return dino_embs @ q_dino

    if mode == "CLIP":
        return clip_embs @ q_clip

    if mode == "Fusion DINO+CLIP":
        s_dino = dino_embs @ q_dino
        s_clip = clip_embs @ q_clip
        return alpha * s_dino + (1.0 - alpha) * s_clip

    raise ValueError(f"Unknown mode: {mode}")


def apply_rocchio_feedback(
    q_current: np.ndarray,
    positive_vectors: list[np.ndarray],
    negative_vectors: list[np.ndarray],
    beta: float,
    gamma: float,
    delta: float,
):
    q_new = beta * q_current.copy()

    if len(positive_vectors) > 0:
        pos_mean = np.mean(np.stack(positive_vectors, axis=0), axis=0)
        q_new += gamma * pos_mean

    if len(negative_vectors) > 0:
        neg_mean = np.mean(np.stack(negative_vectors, axis=0), axis=0)
        q_new -= delta * neg_mean

    return normalize(q_new.astype(np.float32))


def reset_feedback_state():
    for key in [
        "query_initialized",
        "q_dino_current",
        "q_clip_current",
        "q_dino_initial",
        "q_clip_initial",
        "positive_paths",
        "negative_paths",
        "seen_paths",
        "iteration",
    ]:
        if key in st.session_state:
            del st.session_state[key]


st.title("Interactive CBIR with relevance feedback")

left, right = st.columns([1, 2], gap="large")

with left:
    dino_dump = st.text_input("Dump DINO", value="dump_dino.pk1")
    clip_dump = st.text_input("Dump CLIP", value="dump_clip.pk1")

    mode = st.radio(
        "Mode de recherche",
        ["DINO", "CLIP", "Fusion DINO+CLIP"],
        index=2,
    )

    alpha = st.slider(
        "α — poids DINO dans la fusion",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
    )

    topk = st.slider("Nombre de résultats affichés", 5, 30, 10)

    st.markdown("### Relevance feedback")

    beta = st.slider(
        "β — poids de la requête courante",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )

    gamma = st.slider(
        "γ — poids des images pertinentes",
        min_value=0.0,
        max_value=2.0,
        value=0.8,
        step=0.1,
    )

    delta = st.slider(
        "δ — poids des images non pertinentes",
        min_value=0.0,
        max_value=2.0,
        value=0.4,
        step=0.1,
    )

    hide_seen = st.checkbox("Masquer les images déjà annotées", value=True)

    text_prompt = st.text_input("Prompt texte optionnel pour CLIP", value="")
    text_weight = st.slider("Poids du texte CLIP", 0.0, 2.0, 0.7, 0.05)

    uploaded = st.file_uploader("Image requête", type=["jpg", "jpeg", "png"])

    if st.button("Réinitialiser la recherche"):
        reset_feedback_state()
        st.rerun()


with right:
    if not os.path.exists(dino_dump):
        st.error(f"Dump DINO introuvable : {dino_dump}")
        st.stop()

    dino_paths, dino_embs = load_dump_matrix(dino_dump)

    clip_paths = None
    clip_embs = None
    use_clip = mode in ["CLIP", "Fusion DINO+CLIP"]

    if use_clip:
        if not os.path.exists(clip_dump):
            st.error(f"Dump CLIP introuvable : {clip_dump}")
            st.stop()

        raw_clip_paths, raw_clip_embs = load_dump_matrix(clip_dump)
        aligned_paths, dino_keep_idx, clip_embs = align_clip_to_dino_paths(
            dino_paths,
            raw_clip_paths,
            raw_clip_embs,
        )

        dino_paths = aligned_paths
        dino_embs = dino_embs[dino_keep_idx]

    dino_device, dino_processor, dino_model = load_dino()

    if use_clip:
        clip_device, clip_processor, clip_model = load_clip()

    if uploaded is None:
        st.info("Charge une image pour lancer la recherche.")
        st.stop()

    query_img = Image.open(uploaded).convert("RGB")

    st.subheader("Image requête")
    st.image(query_img, width=350)

    if "query_initialized" not in st.session_state:
        q_dino = dino_embed(query_img, dino_device, dino_processor, dino_model)

        q_clip = None
        if use_clip:
            q_clip_img = clip_image_embed(query_img, clip_device, clip_processor, clip_model)

            if text_prompt.strip():
                q_text = clip_text_embed(text_prompt.strip(), clip_device, clip_processor, clip_model)
                q_clip = normalize(q_clip_img + text_weight * q_text)
            else:
                q_clip = q_clip_img

        st.session_state.query_initialized = True
        st.session_state.q_dino_initial = q_dino
        st.session_state.q_dino_current = q_dino

        st.session_state.q_clip_initial = q_clip
        st.session_state.q_clip_current = q_clip

        st.session_state.positive_paths = []
        st.session_state.negative_paths = []
        st.session_state.seen_paths = []
        st.session_state.iteration = 0

    q_dino_current = st.session_state.q_dino_current
    q_clip_current = st.session_state.q_clip_current

    scores = compute_scores(
        mode=mode,
        alpha=alpha,
        dino_embs=dino_embs,
        clip_embs=clip_embs,
        q_dino=q_dino_current,
        q_clip=q_clip_current,
    )

    if hide_seen and len(st.session_state.seen_paths) > 0:
        seen_set = set(st.session_state.seen_paths)
        for i, p in enumerate(dino_paths):
            if p in seen_set:
                scores[i] = -np.inf

    top_idx = np.argsort(-scores)[:topk]
    results = [(dino_paths[i], float(scores[i]), int(i)) for i in top_idx]

    st.subheader(f"Résultats — itération {st.session_state.iteration}")

    st.caption(
        "Coche les images pertinentes et/ou non pertinentes, puis clique sur "
        "« Raffiner la recherche »."
    )

    cols = st.columns(5)
    selected_positive = []
    selected_negative = []

    for rank, (path, score, idx) in enumerate(results, start=1):
        col = cols[(rank - 1) % 5]

        try:
            img = Image.open(path).convert("RGB")
            col.image(
                img,
                caption=f"{rank}. score={score:.3f}\n{Path(path).name}",
                use_container_width=True,
            )
        except Exception:
            col.write(f"{rank}. {Path(path).name}")
            col.write(f"score={score:.3f}")

        pos = col.checkbox(
            "Pertinent",
            key=f"pos_{st.session_state.iteration}_{rank}_{path}",
        )
        neg = col.checkbox(
            "Non pertinent",
            key=f"neg_{st.session_state.iteration}_{rank}_{path}",
        )

        if pos and neg:
            col.warning("Choisis un seul label.")
        elif pos:
            selected_positive.append((path, idx))
        elif neg:
            selected_negative.append((path, idx))

    if st.button("Raffiner la recherche"):
        if len(selected_positive) == 0 and len(selected_negative) == 0:
            st.warning("Sélectionne au moins une image pertinente ou non pertinente.")
            st.stop()

        pos_indices = [idx for _, idx in selected_positive]
        neg_indices = [idx for _, idx in selected_negative]

        st.session_state.positive_paths.extend([p for p, _ in selected_positive])
        st.session_state.negative_paths.extend([p for p, _ in selected_negative])
        st.session_state.seen_paths.extend([p for p, _ in selected_positive])
        st.session_state.seen_paths.extend([p for p, _ in selected_negative])

        pos_dino = [dino_embs[i] for i in pos_indices]
        neg_dino = [dino_embs[i] for i in neg_indices]

        st.session_state.q_dino_current = apply_rocchio_feedback(
            q_current=st.session_state.q_dino_current,
            positive_vectors=pos_dino,
            negative_vectors=neg_dino,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )

        if use_clip:
            pos_clip = [clip_embs[i] for i in pos_indices]
            neg_clip = [clip_embs[i] for i in neg_indices]

            st.session_state.q_clip_current = apply_rocchio_feedback(
                q_current=st.session_state.q_clip_current,
                positive_vectors=pos_clip,
                negative_vectors=neg_clip,
                beta=beta,
                gamma=gamma,
                delta=delta,
            )

        st.session_state.iteration += 1
        st.rerun()

    st.markdown("---")

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.subheader("Images pertinentes sélectionnées")
        if len(st.session_state.positive_paths) == 0:
            st.caption("Aucune pour l’instant.")
        else:
            for p in st.session_state.positive_paths:
                st.caption(Path(p).name)

    with col_neg:
        st.subheader("Images non pertinentes sélectionnées")
        if len(st.session_state.negative_paths) == 0:
            st.caption("Aucune pour l’instant.")
        else:
            for p in st.session_state.negative_paths:
                st.caption(Path(p).name)