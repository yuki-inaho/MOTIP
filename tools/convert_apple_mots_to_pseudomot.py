from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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


def _copy_or_link(src: Path, dst: Path, link_images: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_images:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def _image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def _instance_files(path: Path) -> dict[str, Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return {
        item.stem: item
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    }


def _encoded_category(encoded_id: int) -> int:
    category = encoded_id // 1000
    if category <= 0:
        raise ValueError(
            f"AppleMOTS instance value must use MOTS encoding category*1000+instance_id; got {encoded_id}"
        )
    return category


def _mask_rows(
    mask_path: Path,
    frame_id: int,
    track_id_map: dict[int, int],
    compact_track_ids: bool,
    min_area: int,
) -> tuple[list[tuple[int, int, float, float, float, float, float, int, float]], dict[str, Any]]:
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise ValueError(f"AppleMOTS instance mask must be single-channel: {mask_path} shape={mask.shape}")

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [], {"num_instances": 0, "num_skipped_small": 0, "max_encoded_id": 0}

    encoded_values = mask[ys, xs].astype(np.int64)
    order = np.argsort(encoded_values, kind="stable")
    encoded_values = encoded_values[order]
    xs = xs[order]
    ys = ys[order]

    rows: list[tuple[int, int, float, float, float, float, float, int, float]] = []
    num_skipped_small = 0
    start = 0
    while start < len(encoded_values):
        encoded_id = int(encoded_values[start])
        end = start + 1
        while end < len(encoded_values) and int(encoded_values[end]) == encoded_id:
            end += 1

        obj_xs = xs[start:end]
        obj_ys = ys[start:end]
        area = int(end - start)
        if area < min_area:
            num_skipped_small += 1
            start = end
            continue

        category = _encoded_category(encoded_id)
        if compact_track_ids:
            track_id = track_id_map.setdefault(encoded_id, len(track_id_map) + 1)
        else:
            track_id = encoded_id

        x_min = int(obj_xs.min())
        x_max = int(obj_xs.max())
        y_min = int(obj_ys.min())
        y_max = int(obj_ys.max())
        width = float(x_max - x_min + 1)
        height = float(y_max - y_min + 1)
        bbox_area = width * height
        visibility = float(area / bbox_area) if bbox_area > 0 else 0.0
        rows.append((frame_id, track_id, float(x_min), float(y_min), width, height, 1.0, category, visibility))
        start = end

    return rows, {
        "num_instances": len(rows),
        "num_skipped_small": num_skipped_small,
        "max_encoded_id": int(encoded_values[-1]),
    }


def _convert_sequence(
    applemots_root: Path,
    output_root: Path,
    split: str,
    sequence_name: str,
    frame_rate: int,
    link_images: bool,
    compact_track_ids: bool,
    min_area: int,
) -> dict[str, Any]:
    image_dir = applemots_root / split / "images" / sequence_name
    instance_dir = applemots_root / split / "instances" / sequence_name
    images = _image_files(image_dir)
    instances = _instance_files(instance_dir)
    if not images:
        raise ValueError(f"AppleMOTS sequence has no images: {image_dir}")

    sequence_dir = output_root / split / sequence_name
    img1_dir = sequence_dir / "img1"
    gt_dir = sequence_dir / "gt"
    img1_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[int, int, float, float, float, float, float, int, float]] = []
    track_id_map: dict[int, int] = {}
    frame_object_counts: list[int] = []
    num_skipped_small = 0
    width = height = None

    image_stems = {image.stem for image in images}
    extra_instance_files = sorted(stem for stem in instances if stem not in image_stems)
    missing_instance_files: list[str] = []
    for output_frame_id, image_path in enumerate(images, start=1):
        instance_path = instances.get(image_path.stem)
        if instance_path is None:
            missing_instance_files.append(image_path.name)
            continue

        dst = img1_dir / f"{output_frame_id:08d}.jpg"
        _copy_or_link(image_path, dst, link_images=link_images)
        if width is None or height is None:
            with Image.open(image_path) as image:
                width, height = image.size

        frame_rows, frame_summary = _mask_rows(
            mask_path=instance_path,
            frame_id=output_frame_id,
            track_id_map=track_id_map,
            compact_track_ids=compact_track_ids,
            min_area=min_area,
        )
        rows.extend(frame_rows)
        frame_object_counts.append(len(frame_rows))
        num_skipped_small += int(frame_summary["num_skipped_small"])

    if missing_instance_files:
        raise FileNotFoundError(
            f"AppleMOTS sequence {split}/{sequence_name} has images without matching instance masks: "
            f"{missing_instance_files[:5]}"
        )
    if width is None or height is None:
        raise ValueError(f"Could not determine image size for {split}/{sequence_name}")

    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    gt_path = gt_dir / "gt.txt"
    gt_path.write_text(
        "\n".join(
            f"{frame},{track},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{score:.4f},{category},{visibility:.3f}"
            for frame, track, x, y, w, h, score, category, visibility in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )
    seqinfo_path = sequence_dir / "seqinfo.ini"
    _write_seqinfo(seqinfo_path, sequence_name, int(width), int(height), len(images), frame_rate)

    encoded_ids = sorted(track_id_map) if compact_track_ids else sorted({row[1] for row in rows})
    return {
        "split": split,
        "sequence_name": sequence_name,
        "num_frames": len(images),
        "num_objects": len(rows),
        "num_tracks": len(set(row[1] for row in rows)),
        "num_empty_frames": sum(1 for count in frame_object_counts if count == 0),
        "max_objects_per_frame": max(frame_object_counts) if frame_object_counts else 0,
        "num_extra_instance_files": len(extra_instance_files),
        "extra_instance_files": extra_instance_files[:20],
        "num_skipped_small": num_skipped_small,
        "image_width": int(width),
        "image_height": int(height),
        "frame_rate": frame_rate,
        "gt_path": str(gt_path),
        "seqinfo_path": str(seqinfo_path),
        "link_images": link_images,
        "compact_track_ids": compact_track_ids,
        "encoded_id_min": int(encoded_ids[0]) if encoded_ids else None,
        "encoded_id_max": int(encoded_ids[-1]) if encoded_ids else None,
    }


def convert_apple_mots_to_pseudomot(
    applemots_root: str | Path,
    output_root: str | Path,
    splits: list[str],
    frame_rate: int = 30,
    link_images: bool = True,
    compact_track_ids: bool = True,
    min_area: int = 1,
) -> dict[str, Any]:
    applemots_root = Path(applemots_root)
    output_root = Path(output_root)
    if not applemots_root.is_dir():
        raise FileNotFoundError(applemots_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sequence_summaries = []
    for split in splits:
        images_root = applemots_root / split / "images"
        instances_root = applemots_root / split / "instances"
        if not images_root.is_dir() or not instances_root.is_dir():
            raise FileNotFoundError(f"AppleMOTS split must contain images/ and instances/: {split}")
        sequence_names = sorted(item.name for item in images_root.iterdir() if item.is_dir())
        if not sequence_names:
            raise ValueError(f"AppleMOTS split has no sequence directories: {images_root}")
        for sequence_name in sequence_names:
            sequence_summaries.append(
                _convert_sequence(
                    applemots_root=applemots_root,
                    output_root=output_root,
                    split=split,
                    sequence_name=sequence_name,
                    frame_rate=frame_rate,
                    link_images=link_images,
                    compact_track_ids=compact_track_ids,
                    min_area=min_area,
                )
            )

    by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for summary in sequence_summaries:
        split = str(summary["split"])
        by_split[split]["num_sequences"] += 1
        by_split[split]["num_frames"] += int(summary["num_frames"])
        by_split[split]["num_objects"] += int(summary["num_objects"])
        by_split[split]["num_tracks"] += int(summary["num_tracks"])
        by_split[split]["num_empty_frames"] += int(summary["num_empty_frames"])

    summary = {
        "applemots_root": str(applemots_root),
        "output_root": str(output_root),
        "splits": splits,
        "frame_rate": frame_rate,
        "link_images": link_images,
        "compact_track_ids": compact_track_ids,
        "min_area": min_area,
        "by_split": {split: dict(values) for split, values in sorted(by_split.items())},
        "sequences": sequence_summaries,
    }
    (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AppleMOTS instance masks to MOTChallenge/PseudoMOT.")
    parser.add_argument("--applemots-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train"], help="AppleMOTS splits to convert, e.g. train testing.")
    parser.add_argument("--frame-rate", default=30, type=int)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them.")
    parser.add_argument(
        "--preserve-encoded-track-ids",
        action="store_true",
        help="Use raw MOTS pixel values as track IDs instead of compact per-sequence IDs.",
    )
    parser.add_argument("--min-area", default=1, type=int, help="Drop mask instances smaller than this pixel area.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_apple_mots_to_pseudomot(
        applemots_root=args.applemots_root,
        output_root=args.output_root,
        splits=args.splits,
        frame_rate=args.frame_rate,
        link_images=not args.copy_images,
        compact_track_ids=not args.preserve_encoded_track_ids,
        min_area=args.min_area,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
