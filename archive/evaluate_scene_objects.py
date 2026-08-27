#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / n


def topk_indices(sim_row: np.ndarray, k: int) -> np.ndarray:
    if k >= sim_row.shape[0]:
        return np.argsort(-sim_row)
    idx = np.argpartition(-sim_row, kth=k - 1)[:k]
    return idx[np.argsort(-sim_row[idx])]


def scene_from_parent(parent_path: str, anchor="benchs") -> str:
    p = Path(parent_path)
    parts = list(p.parts)
    low = [x.lower() for x in parts]
    for i, part in enumerate(low):
        if part == anchor or part.startswith(anchor):
            if i + 1 < len(parts):
                return parts[i + 1]
    return p.parent.name


def load_object_dump(obj_dump_path: str):
    with open(obj_dump_path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "embeddings" in payload and "meta" in payload:
        emb_dict = payload["embeddings"]  # crop_path -> vec
        meta = payload["meta"]            # crop_path -> record incl parent_path
    else:
        raise RuntimeError("Object dump invalide: attendu {'embeddings','meta'}.")

    crop_paths = list(emb_dict.keys())
    crop_embs = np.stack([emb_dict[p] for p in crop_paths], axis=0).astype(np.float32)
    crop_embs = l2_normalize(crop_embs)

    # group crops by parent image
    parent_to_indices = {}
    parents = []
    for i, cp in enumerate(crop_paths):
        rec = meta.get(cp, {})
        parent = rec.get("parent_path", None)
        if parent is None:
            continue
        parent = os.path.normpath(parent)
        parent_to_indices.setdefault(parent, []).append(i)

    parents = sorted(parent_to_indices.keys())
    return crop_paths, crop_embs, meta, parents, parent_to_indices


def pool_image_embedding(crop_embs: np.ndarray, idxs: list[int]) -> np.ndarray:
    # mean pooling then normalize
    v = crop_embs[idxs].mean(axis=0, keepdims=True).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.squeeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj_dump", type=str, required=True, help="dump_obj_*.pk1 (payload embeddings+meta)")
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="eval_scene_objects_out")
    ap.add_argument("--anchor", type=str, default="benchs", help="Folder name to locate scene label in parent_path")
    args = ap.parse_args()

    if not os.path.exists(args.obj_dump):
        raise FileNotFoundError(args.obj_dump)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_paths, crop_embs, meta, parents, parent_to_indices = load_object_dump(args.obj_dump)
    if len(parents) == 0:
        raise RuntimeError("Aucune image parent trouvée dans le meta. Vérifie le dump_obj.")

    # build one vector per parent image
    img_embs = []
    img_labels = []
    for parent in parents:
        idxs = parent_to_indices[parent]
        if len(idxs) == 0:
            continue
        img_embs.append(pool_image_embedding(crop_embs, idxs))
        img_labels.append(scene_from_parent(parent, anchor=args.anchor))

    img_embs = np.stack(img_embs, axis=0).astype(np.float32)
    img_embs = l2_normalize(img_embs)
    labels = np.array(img_labels)

    # stats
    uniq = sorted(set(labels.tolist()))
    counts = {u: int((labels == u).sum()) for u in uniq}
    print("Scene counts:")
    for u in uniq:
        print(f" - {u}: {counts[u]}")

    # queries: only scenes with >=2 images
    valid = np.where(np.array([counts[l] for l in labels]) > 1)[0]
    if len(valid) == 0:
        raise RuntimeError("Aucune scène avec >=2 images (impossible de calculer recall).")

    rng = np.random.default_rng(args.seed)
    q_idx = valid  # ici: on prend toutes les images comme query (stable sur petit dataset)
    Qn = len(q_idx)

    # similarity
    sim = img_embs[q_idx] @ img_embs.T
    sim[np.arange(Qn), q_idx] = -np.inf

    kmax = min(args.kmax, img_embs.shape[0] - 1)
    top_idx = np.empty((Qn, kmax), dtype=np.int32)
    for i in range(Qn):
        top_idx[i] = topk_indices(sim[i], kmax)

    q_labels = labels[q_idx]
    hits = (labels[top_idx] == q_labels[:, None])
    cum_hits = np.cumsum(hits, axis=1)

    # positives per query
    pos_total = np.array([counts[l] - 1 for l in q_labels], dtype=np.float32)

    ks = np.arange(1, kmax + 1, dtype=np.float32)
    precision = (cum_hits / ks[None, :]).mean(axis=0)
    recall = (cum_hits / pos_total[:, None]).mean(axis=0)

    precision_per_q = cum_hits / ks[None, :]
    ap_per_q = (precision_per_q * hits).sum(axis=1) / pos_total
    mAP = float(ap_per_q.mean())

    print(f"mAP@{kmax}: {mAP:.4f}")
    print(f"P@{kmax}: {precision[-1]:.4f} | R@{kmax}: {recall[-1]:.4f}")

    pd.DataFrame({"k": np.arange(1, kmax + 1), "precision_at_k": precision, "recall_at_k": recall}).to_csv(
        out_dir / "precision_recall_at_k.csv", index=False
    )

    plt.figure()
    plt.plot(np.arange(1, kmax + 1), precision)
    plt.xlabel("K"); plt.ylabel("Precision@K")
    plt.title(f"Scene Precision@K (objects pooled) | Q={Qn}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_at_k.png", dpi=200)

    plt.figure()
    plt.plot(np.arange(1, kmax + 1), recall)
    plt.xlabel("K"); plt.ylabel("Recall@K")
    plt.title(f"Scene Recall@K (objects pooled) | Q={Qn}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "recall_at_k.png", dpi=200)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"obj_dump: {args.obj_dump}\n")
        f.write(f"N_images: {img_embs.shape[0]}\n")
        f.write(f"D: {img_embs.shape[1]}\n")
        f.write(f"queries: {Qn}\n")
        f.write(f"kmax: {kmax}\n")
        f.write(f"mAP@{kmax}: {mAP}\n")
        f.write(f"P@{kmax}: {float(precision[-1])}\n")
        f.write(f"R@{kmax}: {float(recall[-1])}\n")

    print("Saved outputs to:", out_dir.resolve())
    print("Files: precision_recall_at_k.csv, precision_at_k.png, recall_at_k.png, summary.txt")


if __name__ == "__main__":
    main()