#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def canon_path(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def load_pickle(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Le fichier {path} ne contient pas un dictionnaire path -> embedding.")

    return data


def normalize_matrix(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def build_common_index(
    dino_dump: Dict[str, np.ndarray],
    clip_dump: Dict[str, np.ndarray],
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, str]]:
    """
    Retourne les chemins communs aux deux dumps, les matrices DINO/CLIP alignées,
    et un mapping chemin canonique -> chemin original.
    """
    dino_by_canon = {canon_path(k): k for k in dino_dump.keys()}
    clip_by_canon = {canon_path(k): k for k in clip_dump.keys()}

    common_keys = sorted(set(dino_by_canon.keys()) & set(clip_by_canon.keys()))

    if not common_keys:
        raise RuntimeError("Aucune image commune entre les dumps DINOv2 et CLIP.")

    original_paths = {k: dino_by_canon[k] for k in common_keys}

    dino_mat = np.stack([np.asarray(dino_dump[dino_by_canon[k]]) for k in common_keys])
    clip_mat = np.stack([np.asarray(clip_dump[clip_by_canon[k]]) for k in common_keys])

    dino_mat = normalize_matrix(dino_mat)
    clip_mat = normalize_matrix(clip_mat)

    return common_keys, dino_mat, clip_mat, original_paths


def find_query_key(query_path: Optional[str], common_keys: List[str]) -> str:
    if query_path is not None:
        q_key = canon_path(query_path)
        if q_key not in set(common_keys):
            raise ValueError(
                "L'image requête n'est pas présente dans les deux dumps. "
                f"Chemin reçu : {query_path}"
            )
        return q_key

    # Choix automatique : prend une image au milieu du corpus.
    # Tu peux remplacer par un chemin explicite avec --query.
    return common_keys[len(common_keys) // 2]


def topk_indices(scores: np.ndarray, q_idx: int, k: int) -> List[int]:
    scores = scores.copy()
    scores[q_idx] = -np.inf
    idx = np.argsort(-scores)[:k]
    return idx.tolist()


def compute_results(
    q_idx: int,
    dino_mat: np.ndarray,
    clip_mat: np.ndarray,
    alpha: float,
    topk: int,
) -> Dict[str, List[int]]:
    dino_scores = dino_mat @ dino_mat[q_idx]
    clip_scores = clip_mat @ clip_mat[q_idx]
    fusion_scores = alpha * dino_scores + (1.0 - alpha) * clip_scores

    return {
        "DINOv2": topk_indices(dino_scores, q_idx, topk),
        "CLIP": topk_indices(clip_scores, q_idx, topk),
        f"Fusion alpha={alpha:.2f}": topk_indices(fusion_scores, q_idx, topk),
    }


def load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass

    return ImageFont.load_default()


def fit_image(path: str, size: Tuple[int, int]) -> Image.Image:
    """
    Charge une image et la centre dans un canvas blanc de taille fixe.
    """
    canvas = Image.new("RGB", size, "white")

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        # Image manquante ou illisible : placeholder
        img = Image.new("RGB", size, (235, 235, 235))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Image\nillisible", fill="black")
        return img

    img.thumbnail(size, Image.Resampling.LANCZOS)

    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def short_name(path: str, max_len: int = 34) -> str:
    name = Path(path).name
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."


def draw_centered_text(draw: ImageDraw.ImageDraw, box, text: str, font, fill=(0, 0, 0)):
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = x1 + (x2 - x1 - tw) // 2
    y = y1 + (y2 - y1 - th) // 2
    draw.text((x, y), text, font=font, fill=fill)


def make_figure(
    query_path: str,
    results: Dict[str, List[int]],
    common_keys: List[str],
    original_paths: Dict[str, str],
    out_path: Path,
    thumb_size: Tuple[int, int] = (190, 150),
):
    """
    Figure :
    - ligne du haut : image requête
    - ensuite : une ligne DINOv2, une ligne CLIP, une ligne Fusion
    """
    topk = len(next(iter(results.values())))

    margin = 28
    row_gap = 26
    title_h = 42
    label_w = 190
    caption_h = 38

    font_title = load_font(28, bold=True)
    font_method = load_font(23, bold=True)
    font_small = load_font(15, bold=False)

    W = margin * 2 + label_w + topk * thumb_size[0] + (topk - 1) * margin
    query_block_h = title_h + thumb_size[1] + caption_h
    result_row_h = title_h + thumb_size[1] + caption_h

    H = (
        margin
        + query_block_h
        + row_gap
        + len(results) * result_row_h
        + (len(results) - 1) * row_gap
        + margin
    )

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    y = margin

    # -------------------------
    # Requête
    # -------------------------
    draw.text((margin, y), "Image requête", font=font_title, fill="black")
    y += title_h

    q_img = fit_image(query_path, thumb_size)
    q_x = margin + label_w
    canvas.paste(q_img, (q_x, y))
    draw.rectangle([q_x, y, q_x + thumb_size[0], y + thumb_size[1]], outline=(50, 50, 50), width=2)

    draw.text((q_x, y + thumb_size[1] + 8), short_name(query_path), font=font_small, fill=(60, 60, 60))

    y += thumb_size[1] + caption_h + row_gap

    # -------------------------
    # Résultats
    # -------------------------
    for method_name, indices in results.items():
        draw.text((margin, y + 45), method_name, font=font_method, fill="black")

        draw.text(
            (margin, y + 78),
            "Top résultats globaux",
            font=font_small,
            fill=(80, 80, 80),
        )

        x = margin + label_w

        for rank, idx in enumerate(indices, start=1):
            key = common_keys[idx]
            path = original_paths[key]

            img = fit_image(path, thumb_size)
            canvas.paste(img, (x, y + title_h))
            draw.rectangle(
                [x, y + title_h, x + thumb_size[0], y + title_h + thumb_size[1]],
                outline=(70, 70, 70),
                width=2,
            )

            draw.text(
                (x, y + title_h + thumb_size[1] + 6),
                f"#{rank}  {short_name(path, 24)}",
                font=font_small,
                fill=(60, 60, 60),
            )

            x += thumb_size[0] + margin

        y += result_row_h + row_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    print(f"Figure sauvegardée : {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Génère une figure comparant DINOv2, CLIP et la fusion sur une requête globale."
    )

    parser.add_argument("--dino", default="dump_dino_scenes_full.pk1")
    parser.add_argument("--clip", default="dump_clip_scenes_full.pk1")
    parser.add_argument("--query", default=None, help="Chemin de l'image requête. Si absent, choix automatique.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Poids DINOv2 dans la fusion.")
    parser.add_argument("--topk", type=int, default=5, help="Nombre de résultats affichés par méthode.")
    parser.add_argument("--out", default="figures/evaluation/global_dino_clip_fusion_example.png")

    args = parser.parse_args()

    dino_dump = load_pickle(Path(args.dino))
    clip_dump = load_pickle(Path(args.clip))

    common_keys, dino_mat, clip_mat, original_paths = build_common_index(dino_dump, clip_dump)

    q_key = find_query_key(args.query, common_keys)
    q_idx = common_keys.index(q_key)
    query_path = original_paths[q_key]

    print(f"Images communes : {len(common_keys)}")
    print(f"Image requête : {query_path}")

    results = compute_results(
        q_idx=q_idx,
        dino_mat=dino_mat,
        clip_mat=clip_mat,
        alpha=args.alpha,
        topk=args.topk,
    )

    make_figure(
        query_path=query_path,
        results=results,
        common_keys=common_keys,
        original_paths=original_paths,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()