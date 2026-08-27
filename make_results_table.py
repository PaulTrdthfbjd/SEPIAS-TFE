#!/usr/bin/env python3
"""Consolide les ppp_<methode>_<backbone>.csv en un seul tableau (+ LaTeX)."""
import argparse
import glob
import re
from pathlib import Path

import pandas as pd


def parse_name(fname):
    m = re.search(r"ppp_([a-z0-9]+)_(dino|clip)", Path(fname).stem, re.I)
    if m:
        return m.group(1), m.group(2).upper()
    return Path(fname).stem, "?"


def pick_map_col(df):
    return [c for c in df.columns if c.startswith("mAP@")][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="ppp_*.csv")
    ap.add_argument("--out_csv", default="results_summary.csv")
    ap.add_argument("--out_tex", default="results_summary.tex")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"Aucun fichier ne correspond à {args.glob}")

    rows = []
    for f in files:
        seg, backbone = parse_name(f)
        df = pd.read_csv(f)
        map_col = pick_map_col(df)
        sub = df[df["method"].isin({"PRE_global", "POST_max"})].copy()
        fus = df[df["method"].str.startswith("FUSION_b=")]
        if len(fus):
            best = fus.loc[fus["P@10"].idxmax()].copy()
            best["method"] = f"FUSION_best({best['method'].split('=')[1]})"
            sub = pd.concat([sub, best.to_frame().T], ignore_index=True)
        for _, r in sub.iterrows():
            rows.append({
                "segmentation": seg, "backbone": backbone, "variante": r["method"],
                "P@10": round(float(r["P@10"]), 4),
                "R@10": round(float(r["R@10"]), 4),
                map_col: round(float(r[map_col]), 4),
                "HitRate@10": round(float(r["HitRate@10"]), 4),
                "MRR": round(float(r["MRR"]), 4),
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(out.to_latex(index=False, float_format="%.4f"))
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"\nÉcrit : {args.out_csv} et {args.out_tex}")


if __name__ == "__main__":
    main()