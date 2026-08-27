#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_style_label(path_str: str) -> str:
    """
    Extract style label from a path like:
    .../archive/<STYLE>/.../file.jpg

    If 'archive' isn't found in the path parts, fallback to parent folder name.
    """
    p = Path(path_str)
    parts = [x.lower() for x in p.parts]
    if "archive" in parts:
        idx = parts.index("archive")
        if idx + 1 < len(p.parts):
            return p.parts[idx + 1]  # keep original case
    return p.parent.name


def load_dump(dump_path: str):
    with open(dump_path, "rb") as f:
        ref = pickle.load(f)  # dict: path -> embedding (np array)

    paths = list(ref.keys())
    embs = np.stack([ref[p] for p in paths], axis=0).astype(np.float32)  # (N, D)
    return paths, embs


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def topk_indices(sim_row: np.ndarray, k: int) -> np.ndarray:
    """
    Get top-k indices for a 1D similarity array (descending) efficiently.
    """
    if k >= sim_row.shape[0]:
        return np.argsort(-sim_row)
    idx = np.argpartition(-sim_row, kth=k - 1)[:k]
    # sort these k by score
    idx = idx[np.argsort(-sim_row[idx])]
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=str, default="dump_archive.pk1", help="Path to dump file (pickle).")
    ap.add_argument("--kmax", type=int, default=100, help="Max K for curves (Precision@K, Recall@K).")
    ap.add_argument("--n_queries", type=int, default=200, help="Number of query images sampled randomly.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--out_dir", type=str, default="eval_out", help="Output directory.")
    args = ap.parse_args()

    dump_path = args.dump
    if not os.path.exists(dump_path):
        raise FileNotFoundError(f"Dump not found: {dump_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dump: {dump_path}")
    paths, embs = load_dump(dump_path)
    N, D = embs.shape
    print(f"Loaded {N} embeddings of dim {D}")

    # Normalize embeddings for cosine similarity via dot product
    embs = l2_normalize(embs)

    # Build class ids from folder style
    labels = [extract_style_label(p) for p in paths]
    unique_labels = sorted(set(labels))
    label_to_id = {lab: i for i, lab in enumerate(unique_labels)}
    class_ids = np.array([label_to_id[lab] for lab in labels], dtype=np.int32)

    # Class counts (for recall denominator)
    counts = np.bincount(class_ids, minlength=len(unique_labels))
    # Each query has (count-1) positives if we exclude itself
    # Some classes might have count=1 -> recall undefined; we'll avoid selecting them as queries.
    valid_query_mask = counts[class_ids] > 1
    valid_indices = np.where(valid_query_mask)[0]
    if len(valid_indices) == 0:
        raise RuntimeError("No class has more than 1 image -> cannot compute recall meaningfully.")

    rng = np.random.default_rng(args.seed)
    n_queries = min(args.n_queries, len(valid_indices))
    query_idx = rng.choice(valid_indices, size=n_queries, replace=False)
    query_class = class_ids[query_idx]
    query_pos_total = counts[query_class] - 1  # exclude self

    print(f"Sampling {n_queries} queries (classes with >=2 images)")

    # Batch similarity: (Q, D) @ (D, N) => (Q, N)
    Q = embs[query_idx]  # (Q, D)
    sim = Q @ embs.T     # (Q, N)

    # Exclude self match
    sim[np.arange(n_queries), query_idx] = -np.inf

    kmax = args.kmax
    kmax = min(kmax, N - 1)

    # Retrieve top-kmax indices per query
    top_idx = np.empty((n_queries, kmax), dtype=np.int32)
    for i in range(n_queries):
        top_idx[i] = topk_indices(sim[i], kmax)

    # Relevance of retrieved results (proxy ground truth = same class)
    retrieved_class = class_ids[top_idx]  # (Q, K)
    hits = (retrieved_class == query_class[:, None])  # (Q, K) bool

    # Cumulative hits for each k
    cum_hits = np.cumsum(hits, axis=1)  # (Q, K)

    # Precision@k, Recall@k averaged
    ks = np.arange(1, kmax + 1, dtype=np.float32)  # (K,)
    precision = (cum_hits / ks[None, :]).mean(axis=0)  # (K,)
    recall = (cum_hits / query_pos_total[:, None]).mean(axis=0)  # (K,)

    # mAP@kmax
    # AP per query = sum_k (P@k * rel_k) / (#positives)
    precision_per_q = cum_hits / ks[None, :]  # (Q, K)
    ap_per_q = (precision_per_q * hits).sum(axis=1) / query_pos_total
    mAP = float(ap_per_q.mean())

    print(f"mAP@{kmax}: {mAP:.4f}")
    print(f"P@10: {precision[9]:.4f}" if kmax >= 10 else f"P@{kmax}: {precision[-1]:.4f}")
    print(f"R@10: {recall[9]:.4f}" if kmax >= 10 else f"R@{kmax}: {recall[-1]:.4f}")

    # Save curves CSV
    df = pd.DataFrame({
        "k": np.arange(1, kmax + 1),
        "precision_at_k": precision,
        "recall_at_k": recall
    })
    df.to_csv(out_dir / "precision_recall_at_k.csv", index=False)

    # Plot Precision@K and Recall@K
    plt.figure()
    plt.plot(np.arange(1, kmax + 1), precision)
    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.title(f"Precision@K (mean over {n_queries} queries)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_at_k.png", dpi=200)

    plt.figure()
    plt.plot(np.arange(1, kmax + 1), recall)
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title(f"Recall@K (mean over {n_queries} queries)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "recall_at_k.png", dpi=200)

    # Save summary
    summary = {
        "dump": dump_path,
        "N": int(N),
        "D": int(D),
        "n_queries": int(n_queries),
        "kmax": int(kmax),
        "mAP@kmax": mAP,
        "P@10": float(precision[9]) if kmax >= 10 else float(precision[-1]),
        "R@10": float(recall[9]) if kmax >= 10 else float(recall[-1]),
    }
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved outputs to: {out_dir.resolve()}")
    print("Files:")
    print(" - precision_recall_at_k.csv")
    print(" - precision_at_k.png")
    print(" - recall_at_k.png")
    print(" - summary.txt")


if __name__ == "__main__":
    main()