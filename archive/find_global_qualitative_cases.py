#!/usr/bin/env python3
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def canon(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def load_dump(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    d = {canon(k): v for k, v in d.items()}
    paths = list(d.keys())
    embs = np.stack([d[p] for p in paths]).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return paths, embs


def extract_scene_label(path_str: str) -> str:
    p = Path(path_str)
    parts = list(p.parts)
    low = [x.lower() for x in parts]

    for i, part in enumerate(low):
        if part.startswith("benchs"):
            if i + 1 < len(parts):
                return parts[i + 1]

    return p.parent.name


def topk(sim, k, exclude_idx=None):
    sim = sim.copy()
    if exclude_idx is not None:
        sim[exclude_idx] = -np.inf
    idx = np.argsort(-sim)[:k]
    return idx


def make_contact_sheet(query_path, result_rows, out_path, title):
    thumb_w, thumb_h = 180, 150
    margin = 20
    text_h = 55

    cols = 5
    rows = 1 + int(np.ceil(len(result_rows) / cols))

    W = cols * thumb_w + (cols + 1) * margin
    H = rows * (thumb_h + text_h) + (rows + 1) * margin + 30

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 10), title, fill="black")

    def paste_image(path, x, y, label):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            bx = x + (thumb_w - img.width) // 2
            by = y + (thumb_h - img.height) // 2
            canvas.paste(img, (bx, by))
            draw.text((x, y + thumb_h + 3), label[:45], fill="black")
        except Exception as e:
            draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline="red")
            draw.text((x, y), "missing", fill="red")

    # query
    paste_image(query_path, margin, margin + 30, "QUERY")

    start_y = margin + 30 + thumb_h + text_h + margin

    for j, row in enumerate(result_rows):
        p, score, label, ok = row
        r = j // cols
        c = j % cols
        x = margin + c * (thumb_w + margin)
        y = start_y + r * (thumb_h + text_h + margin)

        prefix = "✓" if ok else "×"
        caption = f"{prefix} {j+1}. {score:.3f} | {label}"
        paste_image(p, x, y, caption)

    canvas.save(out_path)


def evaluate_one_query(qi, paths, labels, embs):
    q = embs[qi]
    sims = embs @ q
    idx = topk(sims, 10, exclude_idx=qi)
    hits = [labels[i] == labels[qi] for i in idx]
    p10 = float(np.mean(hits))
    return idx, p10


def main():
    dino_dump = "dump_dino_scenes_full.pk1"
    clip_dump = "dump_clip_scenes_full.pk1"
    out_dir = Path("qualitative_global_cases")
    out_dir.mkdir(exist_ok=True)

    d_paths, d_embs = load_dump(dino_dump)
    c_paths, c_embs = load_dump(clip_dump)

    common = [p for p in d_paths if p in set(c_paths)]
    d_map = {p: i for i, p in enumerate(d_paths)}
    c_map = {p: i for i, p in enumerate(c_paths)}

    paths = common
    d_embs = d_embs[[d_map[p] for p in common]]
    c_embs = c_embs[[c_map[p] for p in common]]

    labels = [extract_scene_label(p) for p in paths]

    alphas = [0.25, 0.5, 0.75]

    candidates = []

    for qi in range(len(paths)):
        label = labels[qi]

        idx_d, p10_d = evaluate_one_query(qi, paths, labels, d_embs)
        idx_c, p10_c = evaluate_one_query(qi, paths, labels, c_embs)

        fusion_scores = {}
        fusion_idx = {}

        for a in alphas:
            sims_f = a * (d_embs @ d_embs[qi]) + (1 - a) * (c_embs @ c_embs[qi])
            idx_f = topk(sims_f, 10, exclude_idx=qi)
            hits_f = [labels[i] == label for i in idx_f]
            p10_f = float(np.mean(hits_f))
            fusion_scores[a] = p10_f
            fusion_idx[a] = idx_f

        best_alpha = max(alphas, key=lambda a: fusion_scores[a])
        best_f = fusion_scores[best_alpha]

        # Cherche les cas où DINO ou fusion bat CLIP.
        if p10_d > p10_c or best_f > p10_c:
            candidates.append({
                "qi": qi,
                "path": paths[qi],
                "label": label,
                "p10_dino": p10_d,
                "p10_clip": p10_c,
                "p10_fusion": best_f,
                "alpha": best_alpha,
                "idx_dino": idx_d,
                "idx_clip": idx_c,
                "idx_fusion": fusion_idx[best_alpha],
            })

    # Trier par gain par rapport à CLIP
    candidates.sort(
        key=lambda x: max(x["p10_dino"], x["p10_fusion"]) - x["p10_clip"],
        reverse=True
    )

    print(f"Found {len(candidates)} candidate queries where DINO or fusion beats CLIP.")

    for rank, cand in enumerate(candidates[:10], start=1):
        print(
            f"{rank:02d} | qi={cand['qi']} | label={cand['label']} | "
            f"DINO P@10={cand['p10_dino']:.2f} | "
            f"CLIP P@10={cand['p10_clip']:.2f} | "
            f"Fusion alpha={cand['alpha']} P@10={cand['p10_fusion']:.2f} | "
            f"{Path(cand['path']).name}"
        )

        for method_name, idxs, p10 in [
            ("dino", cand["idx_dino"], cand["p10_dino"]),
            ("clip", cand["idx_clip"], cand["p10_clip"]),
            ("fusion", cand["idx_fusion"], cand["p10_fusion"]),
        ]:
            rows = []
            for i in idxs:
                s = None
                if method_name == "dino":
                    s = float(d_embs[i] @ d_embs[cand["qi"]])
                elif method_name == "clip":
                    s = float(c_embs[i] @ c_embs[cand["qi"]])
                else:
                    a = cand["alpha"]
                    s = float(a * (d_embs[i] @ d_embs[cand["qi"]]) + (1 - a) * (c_embs[i] @ c_embs[cand["qi"]]))

                rows.append((paths[i], s, labels[i], labels[i] == cand["label"]))

            out_path = out_dir / f"case_{rank:02d}_q{cand['qi']}_{method_name}.jpg"
            make_contact_sheet(
                cand["path"],
                rows,
                out_path,
                title=f"{method_name.upper()} | query label={cand['label']} | P@10={p10:.2f}"
            )

    print(f"Contact sheets saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()