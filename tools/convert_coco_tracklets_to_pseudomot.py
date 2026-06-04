from __future__ import annotations

import argparse
import json
import os
import shutil
from configparser import ConfigParser
from pathlib import Path
from typing import Any


def _load_coco(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ["images", "annotations"]:
        if key not in data:
            raise ValueError(f"COCO file must contain '{key}': {path}")
    return data


def _prepare_output_dirs(output_root: Path, split: str, sequence_name: str) -> dict[str, Path]:
    sequence_dir = output_root / split / sequence_name
    paths = {
        "sequence": sequence_dir,
        "img1": sequence_dir / "img1",
        "gt": sequence_dir / "gt",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _write_seqinfo(path: Path, sequence_name: str, width: int, height: int, length: int, frame_rate: int) -> None:
    parser = ConfigParser()
    parser.optionxform = str
    parser["Sequence"] = {
        "name": sequence_name,
        "imDir": "img1",
        "frameRate": str(frame_rate),
        "seqLength": str(length),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": ".jpg",
    }
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle, space_around_delimiters=False)


def _copy_or_link_image(src: Path, dst: Path, link_images: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_images:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def convert_coco_tracklets_to_pseudomot(
    coco_path: str | Path,
    image_dir: str | Path,
    output_root: str | Path,
    sequence_name: str = "nyx660_jun04",
    split: str = "train",
    frame_rate: int = 30,
    link_images: bool = True,
) -> dict[str, Any]:
    coco_path = Path(coco_path)
    image_dir = Path(image_dir)
    output_root = Path(output_root)
    data = _load_coco(coco_path)
    images = sorted(data["images"], key=lambda item: item["file_name"])
    if not images:
        raise ValueError(f"COCO file has no images: {coco_path}")

    image_by_id = {image["id"]: image for image in images}
    frame_by_image_id = {image["id"]: idx + 1 for idx, image in enumerate(images)}
    width = int(images[0]["width"])
    height = int(images[0]["height"])
    output_paths = _prepare_output_dirs(output_root, split, sequence_name)

    for frame_idx, image in enumerate(images, start=1):
        if int(image["width"]) != width or int(image["height"]) != height:
            raise ValueError("All images must have the same width and height for one MOT sequence")
        src = image_dir / image["file_name"]
        dst = output_paths["img1"] / f"{frame_idx:08d}.jpg"
        _copy_or_link_image(src=src, dst=dst, link_images=link_images)

    rows: list[tuple[int, int, float, float, float, float, float, int, float]] = []
    track_ids: set[int] = set()
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in image_by_id:
            raise ValueError(f"annotation references missing image_id={image_id}")
        attributes = annotation.get("attributes") or {}
        if "track_id" not in attributes:
            raise ValueError(f"annotation id={annotation.get('id')} has no attributes.track_id")
        if "score" not in attributes:
            raise ValueError(f"annotation id={annotation.get('id')} has no attributes.score")
        x, y, w, h = [float(v) for v in annotation["bbox"]]
        if w <= 0 or h <= 0:
            continue
        track_id = int(attributes["track_id"])
        track_ids.add(track_id)
        rows.append(
            (
                frame_by_image_id[image_id],
                track_id + 1,
                x,
                y,
                w,
                h,
                float(attributes["score"]),
                1,
                1.0,
            )
        )
    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))

    gt_path = output_paths["gt"] / "gt.txt"
    gt_path.write_text(
        "\n".join(
            f"{frame},{track},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{score:.3f},{category},{visibility:.3f}"
            for frame, track, x, y, w, h, score, category, visibility in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )
    _write_seqinfo(
        path=output_paths["sequence"] / "seqinfo.ini",
        sequence_name=sequence_name,
        width=width,
        height=height,
        length=len(images),
        frame_rate=frame_rate,
    )

    summary = {
        "coco_path": str(coco_path),
        "image_dir": str(image_dir),
        "output_root": str(output_root),
        "sequence_name": sequence_name,
        "split": split,
        "num_frames": len(images),
        "num_objects": len(rows),
        "num_tracks": len(track_ids),
        "min_frame": rows[0][0] if rows else None,
        "max_frame": rows[-1][0] if rows else None,
        "gt_path": str(gt_path),
        "seqinfo_path": str(output_paths["sequence"] / "seqinfo.ini"),
        "link_images": link_images,
    }
    (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO tracklet annotations to MOTChallenge/PseudoMOT.")
    parser.add_argument("--coco", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sequence-name", default="nyx660_jun04")
    parser.add_argument("--split", default="train")
    parser.add_argument("--frame-rate", default=30, type=int)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_coco_tracklets_to_pseudomot(
        coco_path=args.coco,
        image_dir=args.image_dir,
        output_root=args.output_root,
        sequence_name=args.sequence_name,
        split=args.split,
        frame_rate=args.frame_rate,
        link_images=not args.copy_images,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
