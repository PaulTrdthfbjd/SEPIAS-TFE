#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


METHODS = {
    "maskrcnn": {
        "label": "Mask R-CNN",
        "manifest": "objects_manifest_maskrcnn_scenes_full.jsonl",
    },
    "samauto": {
        "label": "SAM automatique",
        "manifest": "objects_manifest_samauto_scenes_full.jsonl",
    },
    "segnet": {
        "label": "SegNet",
        "manifest": "objects_manifest_segnet_scenes_full.jsonl",
    },
    "unet": {
        "label": "U-Net",
        "manifest": "objects_manifest_unet_scenes_full.jsonl",
    },
}


def canon_path(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def get_parent_path(record: dict) -> Optional[str]:
    for key in ["parent_path", "image_path", "parent", "source_path", "img_path"]:
        if key in record and record[key]:
            return str(record[key])
    return None


def get_bbox(record: dict) -> Optional[Tuple[float, float, float, float]]:
    """
    Expected format: bbox_xyxy = [x1, y1, x2, y2].
    Handles a few fallback names for robustness.
    """
    for key in ["bbox_xyxy", "bbox", "box_xyxy"]:
        if key in record and record[key] is not None:
            b = record[key]
            if isinstance(b, dict):
                if all(k in b for k in ["x1", "y1", "x2", "y2"]):
                    return float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                return float(b[0]), float(b[1]), float(b[2]), float(b[3])

    if all(k in record for k in ["x1", "y1", "x2", "y2"]):
        return (
            float(record["x1"]),
            float(record["y1"]),
            float(record["x2"]),
            float(record["y2"]),
        )

    return None


def get_score(record: dict) -> float:
    for key in ["score", "confidence", "pred_iou", "stability_score"]:
        if key in record and record[key] is not None:
            try:
                return float(record[key])
            except Exception:
                pass
    return 0.0


def load_manifest(manifest_path: Path) -> Dict[str, List[dict]]:
    """
    Returns:
        parent_path_canon -> list of records
    """
    grouped: Dict[str, List[dict]] = {}

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest introuvable: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Ligne JSON invalide ignorée: {manifest_path}:{line_id}")
                continue

            parent = get_parent_path(rec)
            bbox = get_bbox(rec)

            if parent is None or bbox is None:
                continue

            key = canon_path(parent)
            grouped.setdefault(key, []).append(rec)

    return grouped


def bbox_area(b: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def choose_image_auto(grouped_by_method: Dict[str, Dict[str, List[dict]]]) -> str:
    """
    Chooses an image present in all methods.
    Preference: enough boxes in every method, but not absurdly many.
    """
    common = None
    for grouped in grouped_by_method.values():
        keys = set(grouped.keys())
        common = keys if common is None else common & keys

    if not common:
        raise RuntimeError("Aucune image commune trouvée dans les quatre manifests.")

    def score_image(parent_key: str) -> float:
        counts = [len(grouped_by_method[m][parent_key]) for m in METHODS]
        # Prefer images with several regions, but penalize very crowded ones.
        min_count = min(counts)
        total = sum(counts)
        return min_count * 10 - abs(total - 28)

    chosen = max(common, key=score_image)
    return chosen


def resolve_original_path(parent_key: str, grouped_by_method: Dict[str, Dict[str, List[dict]]]) -> Path:
    """
    Retrieve the original path string from one of the records.
    """
    for method in METHODS:
        records = grouped_by_method[method].get(parent_key, [])
        if records:
            parent = get_parent_path(records[0])
            if parent:
                p = Path(parent)
                if p.exists():
                    return p

    # Fallback: parent_key itself may be usable
    p = Path(parent_key)
    if p.exists():
        return p

    raise FileNotFoundError(
        "Impossible de retrouver le fichier image original. "
        "Passe explicitement --image avec le chemin complet de l'image."
    )


def select_records(records: List[dict], max_boxes: int) -> List[dict]:
    """
    Keeps the most useful boxes for display.
    Priority:
    1. higher score if available,
    2. otherwise larger area.
    """
    def key(rec: dict):
        b = get_bbox(rec)
        return (get_score(rec), bbox_area(b) if b else 0.0)

    return sorted(records, key=key, reverse=True)[:max_boxes]


def draw_boxes_on_image(
    img: Image.Image,
    records: List[dict],
    max_boxes: int,
    line_width: int = 7,
) -> Image.Image:
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    selected = select_records(records, max_boxes=max_boxes)

    for rec in selected:
        bbox = get_bbox(rec)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        # Clamp to image bounds
        x1 = max(0, min(out.width - 1, x1))
        y1 = max(0, min(out.height - 1, y1))
        x2 = max(0, min(out.width - 1, x2))
        y2 = max(0, min(out.height - 1, y2))

        draw.rectangle([x1, y1, x2, y2], outline=(220, 30, 30), width=line_width)

    return out


def make_thumbnail(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    tmp = img.copy()
    tmp.thumbnail((size[0], size[1]), Image.Resampling.LANCZOS)
    x = (size[0] - tmp.width) // 2
    y = (size[1] - tmp.height) // 2
    canvas.paste(tmp, (x, y))
    return canvas


def make_panel(
    panels: List[Tuple[str, Image.Image, int]],
    out_path: Path,
    thumb_size: Tuple[int, int] = (520, 390),
) -> None:
    """
    Crée une figure 2x2 avec les quatre méthodes de segmentation.
    panels = [(title, image, number_of_boxes), ...]
    """
    n_cols = 2
    n_rows = 2

    margin = 30
    title_h = 42
    footer_h = 34

    cell_w = thumb_size[0]
    cell_h = title_h + thumb_size[1] + footer_h

    W = n_cols * cell_w + (n_cols + 1) * margin
    H = n_rows * cell_h + (n_rows + 1) * margin

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arial.ttf", 24)
        font_small = ImageFont.truetype("arial.ttf", 17)
    except Exception:
        font_title = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for i, (title, img, n_boxes) in enumerate(panels[:4]):
        row = i // n_cols
        col = i % n_cols

        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)

        # Titre
        draw.text((x, y), title, fill="black", font=font_title)

        # Image
        thumb = make_thumbnail(img, thumb_size)
        canvas.paste(thumb, (x, y + title_h))

        # Footer
        footer = f"{n_boxes} région(s) affichée(s)"
        draw.text(
            (x, y + title_h + thumb_size[1] + 8),
            footer,
            fill="black",
            font=font_small,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    print(f"Figure sauvegardée: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Crée une figure comparative Mask R-CNN / SAM-auto / SegNet / U-Net sur une même image."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Chemin de l'image à utiliser. Si absent, le script choisit automatiquement une image commune aux quatre manifests.",
    )
    parser.add_argument(
        "--out",
        default="figures/method/segmentation_comparison.png",
        help="Chemin de sortie de la figure.",
    )
    parser.add_argument(
        "--max_boxes",
        type=int,
        default=10,
        help="Nombre maximal de boîtes affichées par méthode.",
    )
    parser.add_argument(
        "--base_dir",
        default=".",
        help="Dossier contenant les manifests objects_manifest_*.jsonl.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    grouped_by_method: Dict[str, Dict[str, List[dict]]] = {}
    for method, cfg in METHODS.items():
        manifest_path = base_dir / cfg["manifest"]
        grouped_by_method[method] = load_manifest(manifest_path)
        print(f"{cfg['label']}: {sum(len(v) for v in grouped_by_method[method].values())} régions chargées")

    if args.image is not None:
        parent_key = canon_path(args.image)
        original_path = Path(args.image)
        if not original_path.exists():
            raise FileNotFoundError(f"Image introuvable: {original_path}")
    else:
        parent_key = choose_image_auto(grouped_by_method)
        original_path = resolve_original_path(parent_key, grouped_by_method)

    print(f"Image utilisée: {original_path}")

    original = Image.open(original_path).convert("RGB")

    panels: List[Tuple[str, Image.Image, int]] = []

    for method, cfg in METHODS.items():
        records = grouped_by_method[method].get(parent_key, [])

        if not records:
            print(f"[WARN] Aucune région trouvée pour {cfg['label']} sur cette image.")
            overlay = original.copy()
            n_displayed = 0
        else:
            overlay = draw_boxes_on_image(original, records, max_boxes=args.max_boxes)
            n_displayed = min(len(records), args.max_boxes)

        panels.append((cfg["label"], overlay, n_displayed))

    make_panel(
        panels=panels,
        out_path=Path(args.out),
        thumb_size=(520, 390),
    )


if __name__ == "__main__":
    main()