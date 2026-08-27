#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def canon(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def find_bench_anchor(parts_lower):
    for i, x in enumerate(parts_lower):
        if x.startswith("benchs"):
            return i
    return None


def extract_scene_label(path_str: str) -> str:
    p = Path(path_str)
    parts = list(p.parts)
    low = [x.lower() for x in parts]

    idx = find_bench_anchor(low)
    if idx is not None and idx + 1 < len(parts):
        return parts[idx + 1]

    return p.parent.name


def load_dump(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)

    data = {canon(k): v for k, v in data.items()}
    paths = list(data.keys())
    embs = np.stack([data[p] for p in paths], axis=0).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    return paths, embs


def topk_indices(sim_row: np.ndarray, k: int) -> np.ndarray:
    if k >= sim_row.shape[0]:
        return np.argsort(-sim_row)

    idx = np.argpartition(-sim_row, kth=k - 1)[:k]
    return idx[np.argsort(-sim_row[idx])]


def compute_metrics(sim: np.ndarray, class_ids: np.ndarray, query_idx: np.ndarray, kmax: int):
    qn = len(query_idx)
    q_class = class_ids[query_idx]
    counts = np.bincount(class_ids)
    pos_total = counts[q_class] - 1

    sim = sim.copy()
    sim[np.arange(qn), query_idx] = -np.inf

    top_idx = np.empty((qn, kmax), dtype=np.int32)
    for i in range(qn):
        top_idx[i] = topk_indices(sim[i], kmax)

    hits = class_ids[top_idx] == q_class[:, None]
    cum_hits = np.cumsum(hits, axis=1)

    ks = np.arange(1, kmax + 1, dtype=np.float32)

    precision_at_k = cum_hits / ks[None, :]
    recall_at_k = cum_hits / pos_total[:, None]

    precision = precision_at_k.mean(axis=0)
    recall = recall_at_k.mean(axis=0)

    ap_per_query = (precision_at_k * hits).sum(axis=1) / pos_total
    map_k = float(ap_per_query.mean())

    k10 = min(10, kmax)
    hit_rate_10 = float(hits[:, :k10].any(axis=1).mean())

    has_hit = hits.any(axis=1)
    first_hit = np.argmax(hits, axis=1) + 1
    mrr = float((1.0 / first_hit[has_hit]).mean()) if has_hit.any() else 0.0

    return {
        "P@10": float(precision[k10 - 1]),
        "R@10": float(recall[k10 - 1]),
        f"mAP@{kmax}": map_k,
        "HitRate@10": hit_rate_10,
        "MRR": mrr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino", default="dump_dino_scenes_full.pk1")
    ap.add_argument("--clip", default="dump_clip_scenes_full.pk1")
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--n_queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval_global_fusion.csv")
    args = ap.parse_args()

    dino_paths, dino_embs_raw = load_dump(args.dino)
    clip_paths, clip_embs_raw = load_dump(args.clip)

    clip_set = set(clip_paths)
    common_paths = [p for p in dino_paths if p in clip_set]

    if len(common_paths) == 0:
        raise RuntimeError("Aucune image commune entre les dumps DINO et CLIP.")

    dino_map = {p: i for i, p in enumerate(dino_paths)}
    clip_map = {p: i for i, p in enumerate(clip_paths)}

    dino_idx = [dino_map[p] for p in common_paths]
    clip_idx = [clip_map[p] for p in common_paths]

    paths = common_paths
    dino_embs = dino_embs_raw[dino_idx]
    clip_embs = clip_embs_raw[clip_idx]

    labels = [extract_scene_label(p) for p in paths]
    unique_labels = sorted(set(labels))
    label_to_id = {lab: i for i, lab in enumerate(unique_labels)}
    class_ids = np.array([label_to_id[l] for l in labels], dtype=np.int32)

    counts = np.bincount(class_ids)
    valid = np.where(counts[class_ids] > 1)[0]

    rng = np.random.default_rng(args.seed)
    n_queries = min(args.n_queries, len(valid))
    query_idx = rng.choice(valid, size=n_queries, replace=False)

    kmax = min(args.kmax, len(paths) - 1)

    print(f"N images communes: {len(paths)}")
    print(f"N requêtes: {n_queries}")
    print(f"kmax: {kmax}")
    print("Labels:")
    for lab in unique_labels:
        print(f" - {lab}: {counts[label_to_id[lab]]}")

    q_dino = dino_embs[query_idx]
    q_clip = clip_embs[query_idx]

    sim_dino = q_dino @ dino_embs.T
    sim_clip = q_clip @ clip_embs.T

    rows = []

    rows.append({
        "method": "DINOv2",
        **compute_metrics(sim_dino, class_ids, query_idx, kmax),
    })

    rows.append({
        "method": "CLIP",
        **compute_metrics(sim_clip, class_ids, query_idx, kmax),
    })

    for alpha in [0.25, 0.50, 0.75]:
        sim_fusion = alpha * sim_dino + (1.0 - alpha) * sim_clip
        rows.append({
            "method": f"Fusion alpha={alpha:.2f}",
            **compute_metrics(sim_fusion, class_ids, query_idx, kmax),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print()
    print(df.to_string(index=False))
    print()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()