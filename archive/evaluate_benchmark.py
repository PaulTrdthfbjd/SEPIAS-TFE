#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd


def average_precision(rels: np.ndarray) -> float:
    # rels: binary array in ranked order
    if rels.sum() == 0:
        return 0.0
    cum = np.cumsum(rels)
    precision_at_k = cum / (np.arange(len(rels)) + 1)
    return float((precision_at_k * rels).sum() / rels.sum())


def eval_mode(cands, mode: str, alpha: float = 0.5):
    # Build scores
    scores = []
    for c in cands:
        sd = c.get("score_dino", None)
        sc = c.get("score_clip", None)
        if mode == "dino":
            score = -1e9 if sd is None else sd
        elif mode == "clip":
            score = -1e9 if sc is None else sc
        elif mode == "fusion":
            if sd is None and sc is None:
                score = -1e9
            elif sd is None:
                score = sc
            elif sc is None:
                score = sd
            else:
                score = alpha * sd + (1 - alpha) * sc
        else:
            raise ValueError(mode)
        scores.append(score)

    order = np.argsort(-np.array(scores))
    ranked = [cands[i] for i in order]
    return ranked


def compute_metrics(items, mode, alpha=None, k_list=(1, 5, 10)):
    rows = []
    for item in items:
        cands = item["candidates"]
        ranked = eval_mode(cands, mode, alpha if alpha is not None else 0.5)

        for target in ["style", "content", "local"]:
            rel = np.array([1 if r[f"label_{target}"] else 0 for r in ranked], dtype=np.int32)

            ap = average_precision(rel)
            metrics = {"AP": ap}

            for k in k_list:
                k = min(k, len(rel))
                metrics[f"P@{k}"] = float(rel[:k].mean()) if k > 0 else 0.0

            rows.append({
                "mode": mode if alpha is None else f"fusion_{alpha:.2f}",
                "target": target,
                **metrics
            })

    df = pd.DataFrame(rows)
    out = df.groupby(["mode", "target"], as_index=False).mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="benchmark_annotations.json")
    ap.add_argument("--out", type=str, default="benchmark_eval.csv")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["items"]
    if len(items) == 0:
        raise RuntimeError("No items in benchmark JSON.")

    dfs = []
    dfs.append(compute_metrics(items, mode="dino"))
    dfs.append(compute_metrics(items, mode="clip"))
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        dfs.append(compute_metrics(items, mode="fusion", alpha=a))

    out = pd.concat(dfs, axis=0)
    out.to_csv(args.out, index=False)
    print("Saved:", args.out)
    print(out)


if __name__ == "__main__":
    main()