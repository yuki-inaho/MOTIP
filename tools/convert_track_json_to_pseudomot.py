from __future__ import annotations

import argparse
import json
import os
import shutil
from configparser import ConfigParser
from pathlib import Path
from typing import Any

from PIL import Image


def _load_track_json(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        frames = data.get("frames")
        if not isinstance(frames, list):
            raise ValueError(f"Track JSON dict must contain a frames list: {path}")
        return dict(data.get("meta", {})), frames
    if isinstance(data, list):
        return {}, data
    raise ValueError(f"Unsupported track JSON root type: {type(data).__name__}")


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


def convert_track_json_to_pseudomot(
    track_json: str | Path,
    image_dir: str | Path,
    output_root: str | Path,
    sequence_name: str,
    split: str = "train",
    frame_rate: int = 30,
    link_images: bool = True,
) -> dict[str, Any]:
    track_json = Path(track_json)
    image_dir = Path(image_dir)
    output_root = Path(output_root)
    meta, frames = _load_track_json(track_json)
    if not frames:
        raise ValueError(f"Track JSON has no frames: {track_json}")

    sequence_dir = output_root / split / sequence_name
    img_dir = sequence_dir / "img1"
    gt_dir = sequence_dir / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    width = height = None
    rows: list[tuple[int, int, float, float, float, float, float, int, float]] = []
    track_ids: set[int] = set()
    num_input_tracks = 0
    num_skipped_degenerate = 0

    for output_frame_id, frame in enumerate(frames, start=1):
        file_name = str(frame["file_name"])
        src = image_dir / file_name
        dst = img_dir / f"{output_frame_id:08d}.jpg"
        _copy_or_link(src, dst, link_images=link_images)
        if width is None or height is None:
            with Image.open(src) as image:
                width, height = image.size

        for track in frame.get("tracks", []):
            num_input_tracks += 1
            track_id = int(track["track_id"])
            x, y, w, h = [float(value) for value in track["bbox"]]
            if w <= 0 or h <= 0:
                num_skipped_degenerate += 1
                continue
            score = float(track.get("score", 1.0))
            category = int(track.get("category", 0)) + 1
            track_ids.add(track_id)
            rows.append((output_frame_id, track_id, x, y, w, h, score, category, 1.0))

    if width is None or height is None:
        raise ValueError(f"Could not determine image size from {image_dir}")

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
    _write_seqinfo(seqinfo_path, sequence_name, int(width), int(height), len(frames), frame_rate)

    empty_frames = len(frames) - len({row[0] for row in rows})
    summary = {
        "track_json": str(track_json),
        "image_dir": str(image_dir),
        "output_root": str(output_root),
        "sequence_name": sequence_name,
        "split": split,
        "frame_rate": frame_rate,
        "num_frames": len(frames),
        "num_input_tracks": num_input_tracks,
        "num_skipped_degenerate": num_skipped_degenerate,
        "num_objects": len(rows),
        "num_tracks": len(track_ids),
        "num_empty_frames": empty_frames,
        "min_frame": rows[0][0] if rows else None,
        "max_frame": rows[-1][0] if rows else None,
        "gt_path": str(gt_path),
        "seqinfo_path": str(seqinfo_path),
        "link_images": link_images,
        "source_meta": meta,
    }
    (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MOTIP track JSON to MOTChallenge/PseudoMOT.")
    parser.add_argument("--track-json", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--frame-rate", default=30, type=int)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_track_json_to_pseudomot(
        track_json=args.track_json,
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
