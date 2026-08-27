#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_artist(path_str: str) -> str:
    p = Path(path_str)
    stem = p.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return p.parent.name


def load_dump(dump_path: str):
    with open(dump_path, "rb") as f:
        ref = pickle.load(f)
    paths = list(ref.keys())
    embs = np.stack([ref[p] for p in paths], axis=0).astype(np.float32)
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
    ap.add_argument("--dump", type=str, required=True)
    ap.add_argument("--kmax", type=int, default=100)
    ap.add_argument("--n_queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="eval_artist_out")
    args = ap.parse_args()

    if not os.path.exists(args.dump):
        raise FileNotFoundError(args.dump)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, embs = load_dump(args.dump)
    N, D = embs.shape
    embs = l2_normalize(embs)

    labels = [parse_artist(p) for p in paths]
    uniq = sorted(set(labels))
    lab2id = {l: i for i, l in enumerate(uniq)}
    class_ids = np.array([lab2id[l] for l in labels], dtype=np.int32)
    counts = np.bincount(class_ids, minlength=len(uniq))

    valid = np.where(counts[class_ids] > 1)[0]
    if len(valid) == 0:
        raise RuntimeError("No artist has >=2 images. Check naming convention.")

    rng = np.random.default_rng(args.seed)
    Qn = min(args.n_queries, len(valid))
    q_idx = rng.choice(valid, size=Qn, replace=False)

    q_class = class_ids[q_idx]
    pos_total = counts[q_class] - 1

    sim = embs[q_idx] @ embs.T
    sim[np.arange(Qn), q_idx] = -np.inf

    kmax = min(args.kmax, N - 1)
    top_idx = np.empty((Qn, kmax), dtype=np.int32)
    for i in range(Qn):
        top_idx[i] = topk_indices(sim[i], kmax)

    hits = (class_ids[top_idx] == q_class[:, None])
    cum_hits = np.cumsum(hits, axis=1)

    ks = np.arange(1, kmax + 1, dtype=np.float32)
    precision = (cum_hits / ks[None, :]).mean(axis=0)
    recall = (cum_hits / pos_total[:, None]).mean(axis=0)

    precision_per_q = cum_hits / ks[None, :]
    ap_per_q = (precision_per_q * hits).sum(axis=1) / pos_total
    mAP = float(ap_per_q.mean())

    hit_rate_10 = float(hits[:, : min(10, kmax)].any(axis=1).mean())
    has_hit = hits.any(axis=1)
    first_hit = np.argmax(hits, axis=1) + 1
    mrr = float((1.0 / first_hit[has_hit]).mean()) if has_hit.any() else 0.0

    print(f"N={N}, D={D}, queries={Qn}, kmax={kmax}")
    print(f"mAP@{kmax}: {mAP:.4f} | HitRate@10: {hit_rate_10:.4f} | MRR: {mrr:.4f}")

    pd.DataFrame({"k": np.arange(1, kmax + 1), "precision_at_k": precision, "recall_at_k": recall}).to_csv(
        out_dir / "precision_recall_at_k.csv", index=False
    )
    pd.DataFrame({"artist": uniq, "count": counts}).sort_values("count", ascending=False).to_csv(
        out_dir / "artist_counts.csv", index=False
    )

    plt.figure()
    plt.plot(np.arange(1, kmax + 1), precision)
    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.title(f"Artist-level Precision@K (mean over {Qn} queries)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_at_k.png", dpi=200)

    plt.figure()
    plt.plot(np.arange(1, kmax + 1), recall)
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title(f"Artist-level Recall@K (mean over {Qn} queries)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "recall_at_k.png", dpi=200)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"dump: {args.dump}\nN: {N}\nD: {D}\nqueries: {Qn}\nkmax: {kmax}\n")
        f.write(f"mAP@{kmax}: {mAP}\nHitRate@10: {hit_rate_10}\nMRR: {mrr}\n")

    print("Saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()