#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def canon(p: str) -> str:
    # Windows-safe canonical key (comme tes dumps)
    return os.path.normcase(os.path.normpath(p))


def _find_bench_anchor(parts_lower):
    # accepte "benchs", "benchs_full", etc.
    for i, x in enumerate(parts_lower):
        if x.startswith("benchs"):
            return i
    return None

def is_bench_path(path_str: str) -> bool:
    parts = list(Path(path_str).parts)
    low = [x.lower() for x in parts]
    idx = _find_bench_anchor(low)
    return (idx is not None) and (idx + 1 < len(parts))

def extract_scene_label(path_str: str) -> str:
    parts = list(Path(path_str).parts)
    low = [x.lower() for x in parts]
    idx = _find_bench_anchor(low)
    if idx is not None and idx + 1 < len(parts):
        return parts[idx + 1]
    return Path(path_str).parent.name


def load_dump(dump_path: str):
    with open(dump_path, "rb") as f:
        ref = pickle.load(f)  # dict: path -> embedding
    # canonicalise les clés (important)
    ref2 = {canon(k): v for k, v in ref.items()}
    paths = list(ref2.keys())
    embs = np.stack([ref2[p] for p in paths], axis=0).astype(np.float32)
    return paths, embs


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)


def topk_indices(sim_row: np.ndarray, k: int) -> np.ndarray:
    if k >= sim_row.shape[0]:
        return np.argsort(-sim_row)
    idx = np.argpartition(-sim_row, kth=k - 1)[:k]
    return idx[np.argsort(-sim_row[idx])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=str, required=True, help="Dump embeddings (.pk1)")
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--n_queries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="eval_scene_out")
    ap.add_argument("--benchs_only", action="store_true",
                    help="Si activé: on garde uniquement les images sous cbir-code/benchs/")
    args = ap.parse_args()

    if not os.path.exists(args.dump):
        raise FileNotFoundError(f"Dump not found: {args.dump}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dump: {args.dump}")
    paths, embs = load_dump(args.dump)

    # --- filtre benchmark ---
    if args.benchs_only:
        mask = np.array([is_bench_path(p) for p in paths], dtype=bool)
        paths = [p for p, m in zip(paths, mask) if m]
        embs = embs[mask]
        print(f"Filtered to benchs only: {len(paths)} images")

    if len(paths) < 2:
        raise RuntimeError("Pas assez d’images après filtrage. Vérifie que ton dump contient bien cbir-code/benchs/...")

    # normalize
    embs = l2_normalize(embs)

    # labels = scènes
    labels = [extract_scene_label(p) for p in paths]
    uniq = sorted(set(labels))
    lab2id = {l: i for i, l in enumerate(uniq)}
    class_ids = np.array([lab2id[l] for l in labels], dtype=np.int32)
    counts = np.bincount(class_ids, minlength=len(uniq))

    # stats
    print("Scene counts:")
    for l in uniq:
        print(f" - {l}: {counts[lab2id[l]]}")

    valid = np.where(counts[class_ids] > 1)[0]
    if len(valid) == 0:
        raise RuntimeError("Aucune scène avec >=2 images. (Il te faut au moins 2 images par scène.)")

    rng = np.random.default_rng(args.seed)
    n_queries = min(args.n_queries, len(valid))
    q_idx = rng.choice(valid, size=n_queries, replace=False)

    q_class = class_ids[q_idx]
    pos_total = counts[q_class] - 1  # exclure self

    sim = embs[q_idx] @ embs.T
    sim[np.arange(n_queries), q_idx] = -np.inf  # exclude self

    kmax = min(args.kmax, len(paths) - 1)

    top_idx = np.empty((n_queries, kmax), dtype=np.int32)
    for i in range(n_queries):
        top_idx[i] = topk_indices(sim[i], kmax)

    hits = (class_ids[top_idx] == q_class[:, None])  # (Q,K)
    cum_hits = np.cumsum(hits, axis=1)

    hit_rate_10 = float(hits[:, :min(10, kmax)].any(axis=1).mean())

    has_hit = hits.any(axis=1)
    first_hit = np.argmax(hits, axis=1) + 1
    mrr = float((1.0 / first_hit[has_hit]).mean()) if has_hit.any() else 0.0

    ks = np.arange(1, kmax + 1, dtype=np.float32)
    precision = (cum_hits / ks[None, :]).mean(axis=0)
    recall = (cum_hits / pos_total[:, None]).mean(axis=0)

    precision_per_q = cum_hits / ks[None, :]
    ap_per_q = (precision_per_q * hits).sum(axis=1) / pos_total
    mAP = float(ap_per_q.mean())

    print(f"mAP@{kmax}: {mAP:.4f}")
    if kmax >= 10:
        print(f"P@10: {precision[9]:.4f} | R@10: {recall[9]:.4f}")
    else:
        print(f"P@{kmax}: {precision[-1]:.4f} | R@{kmax}: {recall[-1]:.4f}")

    # save csv
    df = pd.DataFrame({"k": np.arange(1, kmax + 1), "precision_at_k": precision, "recall_at_k": recall})
    df.to_csv(out_dir / "precision_recall_at_k.csv", index=False)

    # plots
    plt.figure()
    plt.plot(np.arange(1, kmax + 1), precision)
    plt.xlabel("K"); plt.ylabel("Precision@K")
    plt.title(f"Scene Precision@K (mean over {n_queries} queries)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_at_k.png", dpi=200)

    plt.figure()
    plt.plot(np.arange(1, kmax + 1), recall)
    plt.xlabel("K"); plt.ylabel("Recall@K")
    plt.title(f"Scene Recall@K (mean over {n_queries} queries)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "recall_at_k.png", dpi=200)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"dump: {args.dump}\n")
        f.write(f"N: {len(paths)}\n")
        f.write(f"D: {embs.shape[1]}\n")
        f.write(f"n_queries: {n_queries}\n")
        f.write(f"kmax: {kmax}\n")
        f.write(f"mAP@{kmax}: {mAP}\n")
        f.write(f"HitRate@10: {hit_rate_10}\n")
        f.write(f"MRR: {mrr}\n")
        if kmax >= 10:
            f.write(f"P@10: {float(precision[9])}\nR@10: {float(recall[9])}\n")

    print("Saved outputs to:", out_dir.resolve())
    print("Files: precision_recall_at_k.csv, precision_at_k.png, recall_at_k.png, summary.txt")


if __name__ == "__main__":
    main()