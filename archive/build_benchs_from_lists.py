#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive_root", required=True)
    ap.add_argument("--lists_dir", required=True)
    ap.add_argument("--out_root", default="benchs")
    ap.add_argument("--max_per_scene", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    archive_root = Path(args.archive_root)
    lists_dir = Path(args.lists_dir)
    out_root = Path(args.out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    for txt in sorted(lists_dir.glob("*.txt")):
        scene = txt.stem
        scene_dir = out_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)

        lines = [l.strip() for l in txt.read_text(encoding="utf-8", errors="ignore").splitlines()]
        lines = [l for l in lines if l and not l.startswith("#")]

        if args.max_per_scene and len(lines) > args.max_per_scene:
            lines = lines[:args.max_per_scene]

        n = 0
        for rel in lines:
            rel = rel.replace("\\", "/")
            if rel.startswith("./"):
                rel = rel[2:]
            src = archive_root / rel
            if not src.exists():
                continue

            dst = scene_dir / src.name
            # éviter collisions de noms
            if dst.exists():
                dst = scene_dir / f"{src.stem}__{n}{src.suffix}"

            shutil.copy2(src, dst)
            n += 1

        print(f"{scene}: copied {n} images -> {scene_dir}")

if __name__ == "__main__":
    main()