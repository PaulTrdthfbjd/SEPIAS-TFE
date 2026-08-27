import json
import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Annotate Benchmark", layout="wide")


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


@st.cache_data
def load_dump_matrix(dump_path: str):
    with open(dump_path, "rb") as f:
        ref = pickle.load(f)
    paths = list(ref.keys())
    embs = np.stack([ref[p] for p in paths], axis=0).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return paths, embs


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"schema": "benchmark_v1", "items": []}


st.title("Annotation benchmark (style / contenu / local)")

colA, colB = st.columns([1, 2], gap="large")

with colA:
    dino_dump = st.text_input("Dump DINO", value="dump_dino.pk1")
    clip_dump = st.text_input("Dump CLIP (optionnel)", value="dump_clip.pk1")

    out_json = st.text_input("Fichier sortie JSON", value="benchmark_annotations.json")
    k_dino = st.slider("Nb candidats DINO", 5, 60, 15)
    k_clip = st.slider("Nb candidats CLIP", 5, 60, 15)
    k_rand = st.slider("Nb candidats random", 0, 60, 10)

    query_mode = st.radio("Requête", ["Choisir index corpus", "Uploader image (hors corpus)"], index=0)
    query_index = st.number_input("Index corpus (si applicable)", min_value=0, value=0, step=1)

    st.caption("But : créer un petit corpus annoté pour évaluer l'adaptabilité (style vs contenu vs local).")

with colB:
    if not os.path.exists(dino_dump):
        st.error(f"Dump introuvable: {dino_dump}")
        st.stop()

    d_paths, d_embs = load_dump_matrix(dino_dump)

    use_clip = os.path.exists(clip_dump)
    if use_clip:
        c_paths, c_embs = load_dump_matrix(clip_dump)
        if set(c_paths) != set(d_paths):
            st.warning("DINO/CLIP dumps ne matchent pas, CLIP ignoré.")
            use_clip = False
        else:
            idx_map = {p: i for i, p in enumerate(c_paths)}
            c_embs = np.stack([c_embs[idx_map[p]] for p in d_paths], axis=0)

    # Query
    q_img = None
    q_path = None
    q_dino = None
    q_clip = None

    if query_mode == "Choisir index corpus":
        qi = int(min(max(query_index, 0), len(d_paths) - 1))
        q_path = d_paths[qi]
        q_img = Image.open(q_path).convert("RGB")
        q_dino = d_embs[qi]  # already normalized
        if use_clip:
            q_clip = c_embs[qi]
    else:
        up = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"])
        if up is None:
            st.info("Upload une image pour démarrer.")
            st.stop()
        q_img = Image.open(up).convert("RGB")
        st.warning("Mode upload: annotation limitée aux candidats, sans évaluation globale. "
                   "Recommandé: choisir un index du corpus pour évaluer correctement.")

        # Pour upload, on ne recalcule pas embeddings ici (simple). Si tu veux, je te l’ajoute.
        st.stop()

    st.subheader("Image requête")
    st.image(q_img, use_container_width=True)
    st.caption(q_path)

    # Candidates
    sims_d = d_embs @ q_dino
    sims_d[qi] = -np.inf
    idx_d = np.argsort(-sims_d)[: int(k_dino)]

    cand = {d_paths[i]: {"path": d_paths[i], "score_dino": float(sims_d[i])} for i in idx_d}

    if use_clip:
        sims_c = c_embs @ q_clip
        sims_c[qi] = -np.inf
        idx_c = np.argsort(-sims_c)[: int(k_clip)]
        for i in idx_c:
            p = d_paths[i]
            cand.setdefault(p, {"path": p})
            cand[p]["score_clip"] = float(sims_c[i])

    # Random
    if k_rand > 0:
        rng = np.random.default_rng(42)
        rand_idx = rng.choice(len(d_paths), size=min(int(k_rand), len(d_paths)), replace=False)
        for i in rand_idx:
            p = d_paths[i]
            if p == q_path:
                continue
            cand.setdefault(p, {"path": p})

    candidates = list(cand.values())

    st.subheader("Candidats à annoter")
    st.markdown("Coche : **Style**, **Contenu/Iconographie**, **Local (même objet/partie)**.")
    cols = st.columns(5)
    annotated = []

    for j, item in enumerate(candidates):
        p = item["path"]
        c = cols[j % 5]
        try:
            img = Image.open(p).convert("RGB")
            caption = f"{j+1}. {Path(p).name}\n"
            if "score_dino" in item:
                caption += f"DINO={item['score_dino']:.3f} "
            if "score_clip" in item:
                caption += f"CLIP={item['score_clip']:.3f}"
            c.image(img, caption=caption, use_container_width=True)
        except Exception:
            c.write(Path(p).name)

        style = c.checkbox("Style", key=f"style_{j}")
        content = c.checkbox("Contenu", key=f"content_{j}")
        local = c.checkbox("Local", key=f"local_{j}")

        annotated.append({
            "path": p,
            "score_dino": item.get("score_dino", None),
            "score_clip": item.get("score_clip", None),
            "label_style": style,
            "label_content": content,
            "label_local": local
        })

    if st.button("Enregistrer cette requête"):
        data = load_json(out_json)
        data["items"].append({
            "query_path": q_path,
            "query_index": int(qi),
            "candidates": annotated
        })
        save_json(out_json, data)
        st.success(f"Enregistré. Total requêtes annotées: {len(data['items'])}. Fichier: {out_json}")