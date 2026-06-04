"""Compare two MOTIP tracking JSON files with simple ID proxy metrics.

The input format is the JSON emitted by ``tools/infer_tracklet.py``:

``{"meta": {...}, "frames": [{"frame_id": 1, "tracks": [...]}, ...]}``

This script intentionally avoids ground-truth MOT metrics. It measures whether
track IDs persist longer in the predicted stream, which is the failure mode this
experiment is targeting.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_frames(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    frames = data.get("frames", data) if isinstance(data, dict) else data
    if not isinstance(frames, list):
        raise ValueError(f"Expected a frame list or dict with 'frames': {path}")
    return frames


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0}
    ordered = sorted(values)

    def pick(percentile: float) -> float:
        index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
        return float(ordered[index])

    return {"p50": pick(0.50), "p90": pick(0.90), "p95": pick(0.95)}


def summarize(frames: list[dict[str, Any]]) -> dict[str, Any]:
    detections_per_frame: list[int] = []
    score_values: list[float] = []
    track_lengths: dict[int, int] = {}
    track_first_frame: dict[int, int] = {}
    track_last_frame: dict[int, int] = {}

    for frame in frames:
        frame_id = int(frame.get("frame_id", len(detections_per_frame) + 1))
        tracks = frame.get("tracks", [])
        if not isinstance(tracks, list):
            raise ValueError(f"Frame {frame_id} has non-list 'tracks'")
        detections_per_frame.append(len(tracks))
        for track in tracks:
            track_id = int(track["track_id"])
            track_lengths[track_id] = track_lengths.get(track_id, 0) + 1
            track_first_frame.setdefault(track_id, frame_id)
            track_last_frame[track_id] = frame_id
            if "score" in track:
                score_values.append(float(track["score"]))

    num_frames = len(frames)
    num_detections = sum(detections_per_frame)
    unique_ids = len(track_lengths)
    lengths = list(track_lengths.values())
    spans = [track_last_frame[k] - track_first_frame[k] + 1 for k in track_lengths]

    return {
        "num_frames": num_frames,
        "num_detections": num_detections,
        "unique_track_ids": unique_ids,
        "unique_ids_per_detection": round(unique_ids / num_detections, 6) if num_detections else 0.0,
        "detections_per_frame_mean": round(num_detections / num_frames, 6) if num_frames else 0.0,
        "detections_per_frame_max": max(detections_per_frame, default=0),
        "zero_detection_frames": sum(1 for count in detections_per_frame if count == 0),
        "track_length_mean": round(statistics.fmean(lengths), 6) if lengths else 0.0,
        "track_length_median": float(statistics.median(lengths)) if lengths else 0.0,
        "track_length_max": max(lengths, default=0),
        "track_length_quantiles": quantiles([float(value) for value in lengths]),
        "track_span_mean": round(statistics.fmean(spans), 6) if spans else 0.0,
        "track_span_median": float(statistics.median(spans)) if spans else 0.0,
        "track_span_max": max(spans, default=0),
        "score_mean": round(statistics.fmean(score_values), 6) if score_values else 0.0,
        "score_median": float(statistics.median(score_values)) if score_values else 0.0,
    }


def compare(old_summary: dict[str, Any], new_summary: dict[str, Any]) -> dict[str, Any]:
    old_ratio = float(old_summary["unique_ids_per_detection"])
    new_ratio = float(new_summary["unique_ids_per_detection"])
    old_length = float(old_summary["track_length_mean"])
    new_length = float(new_summary["track_length_mean"])
    return {
        "unique_ids_per_detection_delta": round(new_ratio - old_ratio, 6),
        "unique_ids_per_detection_ratio": round(new_ratio / old_ratio, 6) if old_ratio else None,
        "track_length_mean_delta": round(new_length - old_length, 6),
        "track_length_mean_ratio": round(new_length / old_length, 6) if old_length else None,
        "improved_by_unique_ratio": new_ratio < old_ratio,
        "improved_by_mean_track_length": new_length > old_length,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-json", required=True, type=Path)
    parser.add_argument("--new-json", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "old": summarize(load_frames(args.old_json)),
        "new": summarize(load_frames(args.new_json)),
    }
    result["comparison"] = compare(result["old"], result["new"])
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
        print(f"[compare] JSON -> {args.output_json}")
    print(text)


if __name__ == "__main__":
    main()
