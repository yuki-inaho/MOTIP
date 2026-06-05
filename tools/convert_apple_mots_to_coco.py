from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _copy_or_link(src: Path, dst: Path, link_images: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
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


def _mask_annotations(
    mask_path: Path,
    image_id: int,
    next_annotation_id: int,
    track_id_map: dict[int, int],
    compact_track_ids: bool,
    min_area: int,
    category_id: int,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise ValueError(f"AppleMOTS instance mask must be single-channel: {mask_path} shape={mask.shape}")

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [], next_annotation_id, {"num_instances": 0, "num_skipped_small": 0, "max_encoded_id": 0}

    encoded_values = mask[ys, xs].astype(np.int64)
    order = np.argsort(encoded_values, kind="stable")
    encoded_values = encoded_values[order]
    xs = xs[order]
    ys = ys[order]

    annotations: list[dict[str, Any]] = []
    num_skipped_small = 0
    start = 0
    while start < len(encoded_values):
        encoded_id = int(encoded_values[start])
        end = start + 1
        while end < len(encoded_values) and int(encoded_values[end]) == encoded_id:
            end += 1

        obj_xs = xs[start:end]
        obj_ys = ys[start:end]
        mask_area = int(end - start)
        if mask_area < min_area:
            num_skipped_small += 1
            start = end
            continue

        _encoded_category(encoded_id)
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
        annotations.append(
            {
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x_min), float(y_min), width, height],
                "area": float(width * height),
                "iscrowd": 0,
                "segmentation": [],
                "attributes": {
                    "track_id": track_id,
                    "encoded_id": encoded_id,
                    "mask_area": mask_area,
                    "visibility": float(mask_area / (width * height)) if width > 0 and height > 0 else 0.0,
                },
            }
        )
        next_annotation_id += 1
        start = end

    return annotations, next_annotation_id, {
        "num_instances": len(annotations),
        "num_skipped_small": num_skipped_small,
        "max_encoded_id": int(encoded_values[-1]),
    }


def _convert_split(
    applemots_root: Path,
    output_root: Path,
    split: str,
    link_images: bool,
    compact_track_ids: bool,
    min_area: int,
    category_id: int,
    category_name: str,
    start_image_id: int,
    start_annotation_id: int,
) -> tuple[dict[str, Any], int, int]:
    images_root = applemots_root / split / "images"
    instances_root = applemots_root / split / "instances"
    if not images_root.is_dir() or not instances_root.is_dir():
        raise FileNotFoundError(f"AppleMOTS split must contain images/ and instances/: {split}")

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    sequence_summaries: list[dict[str, Any]] = []
    next_image_id = start_image_id
    next_annotation_id = start_annotation_id

    for sequence_dir in sorted(item for item in images_root.iterdir() if item.is_dir()):
        sequence_name = sequence_dir.name
        image_files = _image_files(sequence_dir)
        instance_files = _instance_files(instances_root / sequence_name)
        image_stems = {image.stem for image in image_files}
        extra_instance_files = sorted(stem for stem in instance_files if stem not in image_stems)
        track_id_map: dict[int, int] = {}
        frame_object_counts: list[int] = []
        num_skipped_small = 0
        width = height = None

        for image_path in image_files:
            instance_path = instance_files.get(image_path.stem)
            if instance_path is None:
                raise FileNotFoundError(f"Missing instance mask for {split}/{sequence_name}/{image_path.name}")
            relative_name = f"{split}/{sequence_name}/{image_path.name}"
            output_image_path = output_root / "images" / relative_name
            _copy_or_link(image_path, output_image_path, link_images=link_images)
            with Image.open(image_path) as image:
                width, height = image.size

            image_id = next_image_id
            next_image_id += 1
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": relative_name,
                    "width": int(width),
                    "height": int(height),
                    "sequence": sequence_name,
                    "split": split,
                    "frame_name": image_path.stem,
                }
            )
            annotations, next_annotation_id, frame_summary = _mask_annotations(
                mask_path=instance_path,
                image_id=image_id,
                next_annotation_id=next_annotation_id,
                track_id_map=track_id_map,
                compact_track_ids=compact_track_ids,
                min_area=min_area,
                category_id=category_id,
            )
            coco_annotations.extend(annotations)
            frame_object_counts.append(len(annotations))
            num_skipped_small += int(frame_summary["num_skipped_small"])

        sequence_summaries.append(
            {
                "split": split,
                "sequence_name": sequence_name,
                "num_frames": len(image_files),
                "num_objects": sum(frame_object_counts),
                "num_tracks": len(track_id_map),
                "num_empty_frames": sum(1 for count in frame_object_counts if count == 0),
                "max_objects_per_frame": max(frame_object_counts) if frame_object_counts else 0,
                "num_extra_instance_files": len(extra_instance_files),
                "extra_instance_files": extra_instance_files[:20],
                "num_skipped_small": num_skipped_small,
                "image_width": int(width or 0),
                "image_height": int(height or 0),
                "compact_track_ids": compact_track_ids,
            }
        )

    coco = {
        "info": {
            "description": "AppleMOTS converted to COCO bbox detection for DEIM/MOTIP experiments",
            "source": str(applemots_root),
            "split": split,
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": category_id, "name": category_name}],
    }
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    ann_path = annotations_dir / f"applemots_{split}.json"
    ann_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    split_summary = {
        "split": split,
        "ann_file": str(ann_path),
        "img_folder": str(output_root / "images"),
        "num_sequences": len(sequence_summaries),
        "num_frames": len(coco_images),
        "num_objects": len(coco_annotations),
        "num_tracks": sum(sequence["num_tracks"] for sequence in sequence_summaries),
        "num_empty_frames": sum(sequence["num_empty_frames"] for sequence in sequence_summaries),
        "sequences": sequence_summaries,
    }
    return split_summary, next_image_id, next_annotation_id


def convert_apple_mots_to_coco(
    applemots_root: str | Path,
    output_root: str | Path,
    splits: list[str],
    link_images: bool = True,
    compact_track_ids: bool = True,
    min_area: int = 1,
    category_id: int = 1,
    category_name: str = "apple",
) -> dict[str, Any]:
    applemots_root = Path(applemots_root)
    output_root = Path(output_root)
    if not applemots_root.is_dir():
        raise FileNotFoundError(applemots_root)
    output_root.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, dict[str, int]] = defaultdict(dict)
    split_summaries: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    for split in splits:
        split_summary, next_image_id, next_annotation_id = _convert_split(
            applemots_root=applemots_root,
            output_root=output_root,
            split=split,
            link_images=link_images,
            compact_track_ids=compact_track_ids,
            min_area=min_area,
            category_id=category_id,
            category_name=category_name,
            start_image_id=next_image_id,
            start_annotation_id=next_annotation_id,
        )
        by_split[split] = {
            "num_sequences": split_summary["num_sequences"],
            "num_frames": split_summary["num_frames"],
            "num_objects": split_summary["num_objects"],
            "num_tracks": split_summary["num_tracks"],
            "num_empty_frames": split_summary["num_empty_frames"],
        }
        split_summaries.append(split_summary)

    summary = {
        "applemots_root": str(applemots_root),
        "output_root": str(output_root),
        "splits": splits,
        "link_images": link_images,
        "compact_track_ids": compact_track_ids,
        "min_area": min_area,
        "category_id": category_id,
        "category_name": category_name,
        "by_split": {split: dict(values) for split, values in sorted(by_split.items())},
        "split_summaries": split_summaries,
        "sequences": [sequence for split_summary in split_summaries for sequence in split_summary["sequences"]],
    }
    (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AppleMOTS instance masks to COCO bbox detection JSON.")
    parser.add_argument("--applemots-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them.")
    parser.add_argument(
        "--preserve-encoded-track-ids",
        action="store_true",
        help="Use raw MOTS pixel values as attributes.track_id instead of compact per-sequence IDs.",
    )
    parser.add_argument("--min-area", default=1, type=int, help="Drop mask instances smaller than this pixel area.")
    parser.add_argument("--category-id", default=1, type=int, help="COCO category id to write. Use 0 for DEIM labels.")
    parser.add_argument("--category-name", default="apple", help="COCO category name to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_apple_mots_to_coco(
        applemots_root=args.applemots_root,
        output_root=args.output_root,
        splits=args.splits,
        link_images=not args.copy_images,
        compact_track_ids=not args.preserve_encoded_track_ids,
        min_area=args.min_area,
        category_id=args.category_id,
        category_name=args.category_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
