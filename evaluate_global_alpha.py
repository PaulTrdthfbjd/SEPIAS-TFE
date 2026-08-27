#!/usr/bin/env python3
"""
Balayage de la fusion alpha (DINOv2 / CLIP) sur la recherche GLOBALE,
en leave-one-out, avec exactement le même protocole que la ligne PRE_global
de evaluate_pre_post_v2.py (mêmes métriques, même kmax, mêmes IC bootstrap).

Fusion au niveau score :  sim = alpha * cos_DINO + (1 - alpha) * cos_CLIP
  alpha = 1.0  ->  DINOv2 seul
  alpha = 0.0  ->  CLIP seul

Entrées : --dino_dump et --clip_dump (dumps globaux {path -> vec}).
Le vivier est l'intersection des chemins présents dans les deux dumps.
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def canon(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def extract_scene_label(path_str: str) -> str:
    parts = list(Path(str(path_str)).parts)
    low = [x.lower() for x in parts]
    idx = next((i for i, x in enumerate(low) if x.startswith("benchs")), None)
    if idx is not None and idx + 1 < len(parts):
        return parts[idx + 1]
    return Path(str(path_str)).parent.name


def l2n(mat):
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)


def load_global(path):
    with open(path, "rb") as f:
        ref = pickle.load(f)
    return {canon(k): np.asarray(v, np.float32) for k, v in ref.items()}


def per_query_stats(scores, cand_labels, q_labels, pos_total, kmax):
    Q = scores.shape[0]
    kmax = min(kmax, scores.shape[1] - 1)
    top = np.empty((Q, kmax), np.int64)
    for i in range(Q):
        row = scores[i]
        idx = np.argpartition(-row, kth=kmax - 1)[:kmax]
        top[i] = idx[np.argsort(-row[idx])]
    hits = (cand_labels[top] == q_labels[:, None])
    cum = np.cumsum(hits, 1)
    ks = np.arange(1, kmax + 1, dtype=np.float32)
    k10 = min(10, kmax)
    return {"P@10": cum[:, k10 - 1] / k10,
            "R@10": cum[:, k10 - 1] / pos_total,
            f"mAP@{kmax}": ((cum / ks[None, :]) * hits).sum(1) / pos_total,
            "HitRate@10": hits[:, :k10].any(1).astype(np.float32),
            "MRR": np.where(hits.any(1), 1.0 / (np.argmax(hits, 1) + 1.0), 0.0)}


def summarize(pq, boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    Q = len(next(iter(pq.values())))
    idxs = rng.integers(0, Q, size=(boot, Q)) if boot else None
    map_key = [k for k in pq if k.startswith("mAP@")][0]
    for k, v in pq.items():
        out[k] = float(v.mean())
        if boot and k in ("P@10", map_key):
            bs = v[idxs].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out[k + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino_dump", required=True)
    ap.add_argument("--clip_dump", required=True)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--alphas", default="0.0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1.0")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default="eval_global_alpha.csv")
    args = ap.parse_args()

    dino = load_global(args.dino_dump)
    clip = load_global(args.clip_dump)
    common = [p for p in dino if p in clip]
    print(f"Images communes DINO∩CLIP : {len(common)}")

    Ed = l2n(np.stack([dino[p] for p in common], 0))
    Ec = l2n(np.stack([clip[p] for p in common], 0))
    labels = np.array([extract_scene_label(p) for p in common])
    uniq, cnt = np.unique(labels, return_counts=True)
    count_of = dict(zip(uniq.tolist(), cnt.tolist()))
    print("Scene counts:", count_of)

    valid = np.array([i for i, l in enumerate(labels) if count_of[l] > 1], np.int64)
    q_labels = labels[valid]
    pos_total = np.array([count_of[l] - 1 for l in q_labels], np.float32)
    Q = len(valid)
    print(f"Requêtes (leave-one-out): {Q}")

    sim_d = Ed[valid] @ Ed.T
    sim_c = Ec[valid] @ Ec.T
    diag = (np.arange(Q), valid)

    rows = []
    for a in [float(x) for x in args.alphas.split(",")]:
        sim = a * sim_d + (1.0 - a) * sim_c
        sim[diag] = -np.inf
        pq = per_query_stats(sim, labels, q_labels, pos_total, args.kmax)
        name = "DINO_seul" if a == 1.0 else ("CLIP_seul" if a == 0.0 else f"alpha={a:g}")
        rows.append({"methode": name, "alpha": a, **summarize(pq, args.bootstrap)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False))
    best = df.loc[df["P@10"].idxmax()]
    print(f"\nMeilleur P@10 : {best['methode']} (alpha={best['alpha']:g}) -> {best['P@10']:.4f}")
    print(f"Écrit : {args.out}")


if __name__ == "__main__":
    main()