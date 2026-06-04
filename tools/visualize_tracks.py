"""Visualize MOTIP tracking results (JSON from tools/infer_tracklet.py) over the source frames.

Reads the tracks JSON + the original image directory, draws each track's bbox and ID
(color keyed by track_id), and writes annotated frames and/or an mp4 video.

No implicit fallback: missing JSON / image dir / frame image raise; at least one of
--output-dir / --output-video must be given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from demo.colormap import get_color


def draw_frame(image, tracks: list[dict], show_score: bool):
    for track in tracks:
        x, y, w, h = (int(round(v)) for v in track["bbox"])
        track_id = int(track["track_id"])
        color = get_color(track_id, rgb=False, use_int=True)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = f"ID {track_id}"
        if show_score and "score" in track:
            label += f" {float(track['score']):.2f}"
        cv2.putText(image, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def run_visualize(args: argparse.Namespace) -> None:
    tracks_json, image_dir = Path(args.tracks_json), Path(args.image_dir)
    for path in (tracks_json, image_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.output_dir and not args.output_video:
        raise SystemExit("Specify --output-dir and/or --output-video.")

    data = json.loads(tracks_json.read_text(encoding="utf-8"))
    frames = data["frames"]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    for frame in frames:
        image_path = image_dir / frame["file_name"]
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise RuntimeError(f"Failed to read frame image: {image_path}")
        draw_frame(bgr, frame["tracks"], args.show_score)
        if args.output_video:
            if writer is None:
                video_path = Path(args.output_video)
                video_path.parent.mkdir(parents=True, exist_ok=True)
                height, width = bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height),
                )
            writer.write(bgr)
        if out_dir:
            cv2.imwrite(str(out_dir / frame["file_name"]), bgr)

    if writer is not None:
        writer.release()
        print(f"[viz] video -> {args.output_video} ({len(frames)} frames @ {args.fps} fps)")
    if out_dir:
        print(f"[viz] {len(frames)} annotated frames -> {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MOTIP tracking JSON over source frames.")
    parser.add_argument("--tracks-json", default="./outputs/tracklet_pseudomot_full/infer/tracks.json")
    parser.add_argument("--image-dir", default="./datasets/TomatoTrackletMOT/train/nyx660_jun04/img1")
    parser.add_argument("--output-dir", default="./outputs/tracklet_pseudomot_full/infer/viz")
    parser.add_argument("--output-video", default=None, help="e.g. ./outputs/.../infer/tracks.mp4")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames.")
    parser.add_argument("--show-score", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_visualize(parse_args())


if __name__ == "__main__":
    main()
