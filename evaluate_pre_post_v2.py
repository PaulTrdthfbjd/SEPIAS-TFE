#!/usr/bin/env python3
"""
Évaluation appariée PRÉ-segmentation (recherche globale) vs POST-segmentation
(recherche hiérarchique crop -> parent), à partir des index .pk1 existants.
Pur numpy, pas de GPU.
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
    ref = {canon(k): np.asarray(v, np.float32) for k, v in ref.items()}
    paths = list(ref.keys())
    return paths, l2n(np.stack([ref[p] for p in paths], 0))


def load_object(path):
    with open(path, "rb") as f:
        pl = pickle.load(f)
    emb, meta = pl["embeddings"], pl["meta"]
    cps = list(emb.keys())
    embs = l2n(np.stack([np.asarray(emb[c], np.float32) for c in cps], 0))
    parent, score, area = [], [], []
    for c in cps:
        r = meta.get(c, {}) if isinstance(meta.get(c, {}), dict) else {}
        parent.append(canon(r.get("parent_path", "")))
        s = r.get("score", None)
        score.append(float(s) if isinstance(s, (int, float)) else np.nan)
        bb = r.get("bbox_xyxy", None)
        if bb and len(bb) == 4:
            area.append(abs((bb[2] - bb[0]) * (bb[3] - bb[1])))
        else:
            area.append(np.nan)
    return embs, np.array(parent), np.array(score, np.float32), np.array(area, np.float32)


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
    p_at10 = cum[:, k10 - 1] / k10
    r_at10 = cum[:, k10 - 1] / pos_total
    ap = ((cum / ks[None, :]) * hits).sum(1) / pos_total
    hit10 = hits[:, :k10].any(1).astype(np.float32)
    has = hits.any(1)
    first = np.argmax(hits, 1) + 1.0
    rr = np.where(has, 1.0 / first, 0.0)
    return {"P@10": p_at10, "R@10": r_at10, f"mAP@{kmax}": ap,
            "HitRate@10": hit10, "MRR": rr}


def summarize(pq, boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    Q = len(next(iter(pq.values())))
    idxs = rng.integers(0, Q, size=(boot, Q)) if boot else None
    map_key = [k for k in pq.keys() if k.startswith("mAP@")][0]
    for k, v in pq.items():
        out[k] = float(v.mean())
        if boot and k in ("P@10", map_key):
            bs = v[idxs].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out[k + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    return out


def local_scores(eq, crop_embs, parent_ids, parent_order, keep_mask, mode="max", topk=3):
    sims = crop_embs @ eq
    out = np.full(len(parent_order), -np.inf, np.float32)
    pos = {p: j for j, p in enumerate(parent_order)}
    buckets = {}
    for ci, par in enumerate(parent_ids):
        if not keep_mask[ci]:
            continue
        j = pos.get(par)
        if j is not None:
            buckets.setdefault(j, []).append(sims[ci])
    for j, lst in buckets.items():
        s = np.array(lst, np.float32)
        if mode == "max":
            out[j] = s.max()
        elif mode == "mean":
            out[j] = s.mean()
        elif mode == "topk":
            out[j] = np.sort(s)[-min(topk, s.shape[0]):].mean()
    return out


def cap_keep_mask(parent_ids, score, area, cap, by):
    if cap <= 0:
        return np.ones(len(parent_ids), bool)
    keep = np.zeros(len(parent_ids), bool)
    order_key = score if by == "score" else area
    key = np.where(np.isnan(order_key), area, order_key)
    key = np.where(np.isnan(key), 0.0, key)
    by_parent = {}
    for i, p in enumerate(parent_ids):
        by_parent.setdefault(p, []).append(i)
    for p, idx in by_parent.items():
        idx = sorted(idx, key=lambda i: key[i], reverse=True)[:cap]
        keep[idx] = True
    return keep


def zscore(col):
    m = np.isfinite(col)
    z = np.full_like(col, -1e9)
    if m.any():
        z[m] = (col[m] - col[m].mean()) / (col[m].std() + 1e-12)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global_dump", required=True)
    ap.add_argument("--obj_dump", required=True)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--cap_per_parent", type=int, default=0)
    ap.add_argument("--cap_by", choices=["score", "area"], default="score")
    ap.add_argument("--restrict_to_covered", action="store_true")
    ap.add_argument("--betas", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--per_class", action="store_true")
    ap.add_argument("--out", default="eval_pre_post_v2.csv")
    args = ap.parse_args()

    g_paths, g_embs = load_global(args.global_dump)
    crop_embs, c_parent, c_score, c_area = load_object(args.obj_dump)
    covered = set(c_parent.tolist())

    if args.restrict_to_covered:
        keep = [i for i, p in enumerate(g_paths) if p in covered]
        g_paths, g_embs = [g_paths[i] for i in keep], g_embs[keep]
        print(f"Restreint aux parents couverts : {len(g_paths)}")

    parents = list(g_paths)
    cand_labels = np.array([extract_scene_label(p) for p in parents])
    uniq, cnt = np.unique(cand_labels, return_counts=True)
    count_of = dict(zip(uniq.tolist(), cnt.tolist()))
    print("Scene counts:", count_of)

    valid = np.array([i for i, l in enumerate(cand_labels) if count_of[l] > 1], np.int64)
    q_labels = cand_labels[valid]
    pos_total = np.array([count_of[l] - 1 for l in q_labels], np.float32)
    Q = len(valid)
    print(f"Requêtes (leave-one-out): {Q} | cap_per_parent={args.cap_per_parent or 'aucun'}")

    keep_mask = cap_keep_mask(c_parent, c_score, c_area, args.cap_per_parent, args.cap_by)
    print(f"Crops retenus après cap : {int(keep_mask.sum())}/{len(c_parent)}")

    sim_pre = g_embs[valid] @ g_embs.T
    sim_pre[np.arange(Q), valid] = -np.inf

    sim_post = {m: np.full((Q, len(parents)), -np.inf, np.float32)
                for m in ("max", "mean", "topk")}
    parent_order = [canon(p) for p in parents]
    for qi, gi in enumerate(valid):
        eq = g_embs[gi]
        for m in ("max", "mean", "topk"):
            row = local_scores(eq, crop_embs, c_parent, parent_order, keep_mask, mode=m, topk=args.topk)
            row[gi] = -np.inf
            sim_post[m][qi] = row

    rows = []

    def add(name, scores):
        pq = per_query_stats(scores, cand_labels, q_labels, pos_total, args.kmax)
        rows.append({"method": name, **summarize(pq, args.bootstrap)})
        return pq

    add("PRE_global", sim_pre)
    for m in ("max", "mean", "topk"):
        add(f"POST_{m}", sim_post[m])
    for beta in [float(b) for b in args.betas.split(",")]:
        fused = np.empty((Q, len(parents)), np.float32)
        for qi in range(Q):
            fused[qi] = beta * zscore(sim_pre[qi]) + (1 - beta) * zscore(sim_post["max"][qi])
        add(f"FUSION_b={beta:g}", fused)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False))
    print(f"\nÉcrit : {args.out}")

    if args.per_class:
        print("\n=== Par classe : PRE_global vs POST_max ===")
        pc = []
        for cls in uniq.tolist():
            sel = np.where(q_labels == cls)[0]
            if len(sel) == 0:
                continue
            a = per_query_stats(sim_pre[sel], cand_labels, q_labels[sel], pos_total[sel], args.kmax)
            b = per_query_stats(sim_post["max"][sel], cand_labels, q_labels[sel], pos_total[sel], args.kmax)
            pc.append({"classe": cls, "n": len(sel),
                       "PRE_P@10": round(float(a["P@10"].mean()), 4),
                       "POST_P@10": round(float(b["P@10"].mean()), 4)})
        print(pd.DataFrame(pc).to_string(index=False))


if __name__ == "__main__":
    main()