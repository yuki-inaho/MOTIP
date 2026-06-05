"""Assign stable track IDs to per-frame detection boxes from MOTIP JSON.

This is a lightweight ByteTrack-style post processor:

1. Treat the input JSON's per-frame ``tracks`` as detections only.
2. Apply optional per-frame NMS.
3. Match high-score detections to active tracks by IoU against a simple
   constant-velocity prediction.
4. Match remaining low-score detections to unmatched tracks.
5. Create new tracks from unmatched high-score detections.

The output uses the same JSON shape as ``tools/infer_tracklet.py`` so existing
visualization and comparison tools can be reused.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def tlwh_to_tlbr(boxes: np.ndarray) -> np.ndarray:
    out = boxes.astype(float, copy=True)
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def iou_matrix(a_tlwh: np.ndarray, b_tlwh: np.ndarray) -> np.ndarray:
    if len(a_tlwh) == 0 or len(b_tlwh) == 0:
        return np.zeros((len(a_tlwh), len(b_tlwh)), dtype=float)
    a = tlwh_to_tlbr(a_tlwh)
    b = tlwh_to_tlbr(b_tlwh)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def center_distance_matrix(a_tlwh: np.ndarray, b_tlwh: np.ndarray) -> np.ndarray:
    if len(a_tlwh) == 0 or len(b_tlwh) == 0:
        return np.zeros((len(a_tlwh), len(b_tlwh)), dtype=float)
    a_centers = np.column_stack((a_tlwh[:, 0] + a_tlwh[:, 2] / 2.0, a_tlwh[:, 1] + a_tlwh[:, 3] / 2.0))
    b_centers = np.column_stack((b_tlwh[:, 0] + b_tlwh[:, 2] / 2.0, b_tlwh[:, 1] + b_tlwh[:, 3] / 2.0))
    return np.linalg.norm(a_centers[:, None, :] - b_centers[None, :, :], axis=2)


def match_by_iou(track_boxes: np.ndarray, det_boxes: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    scores = iou_matrix(track_boxes, det_boxes)
    if scores.size == 0:
        return []
    row_ind, col_ind = linear_sum_assignment(-scores)
    return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if scores[r, c] >= threshold]


def match_by_similarity(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    iou_threshold: float,
    center_distance_thresh: float,
    center_weight: float,
) -> list[tuple[int, int]]:
    ious = iou_matrix(track_boxes, det_boxes)
    if ious.size == 0:
        return []
    if center_distance_thresh <= 0:
        row_ind, col_ind = linear_sum_assignment(-ious)
        return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if ious[r, c] >= iou_threshold]

    distances = center_distance_matrix(track_boxes, det_boxes)
    center_scores = np.exp(-np.square(distances / max(center_distance_thresh, 1e-6)))
    scores = ious + center_weight * center_scores
    valid = (ious >= iou_threshold) | (distances <= center_distance_thresh)
    row_ind, col_ind = linear_sum_assignment(-scores)
    return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if valid[r, c]]


def nms_detections(detections: list[dict[str, Any]], nms_iou: float) -> list[dict[str, Any]]:
    if nms_iou <= 0 or nms_iou >= 1 or len(detections) <= 1:
        return detections
    order = sorted(range(len(detections)), key=lambda i: float(detections[i]["score"]), reverse=True)
    boxes = np.array([d["bbox"] for d in detections], dtype=float)
    keep: list[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        if not order:
            break
        ious = iou_matrix(boxes[[current]], boxes[order])[0]
        order = [idx for idx, iou in zip(order, ious) if iou < nms_iou]
    return [detections[i] for i in keep]


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    score: float
    category: int
    start_frame: int
    last_frame: int
    hits: int = 1
    time_since_update: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    y_start: float = 0.0
    y_last: float = 0.0

    def __post_init__(self) -> None:
        y_center = float(self.bbox[1] + self.bbox[3] / 2.0)
        self.y_start = y_center
        self.y_last = y_center

    def predict_bbox(self, velocity_weight: float) -> np.ndarray:
        predicted = self.bbox + self.velocity * velocity_weight
        predicted[2:] = np.maximum(1.0, predicted[2:])
        return predicted

    def update(self, detection: dict[str, Any], frame_id: int, velocity_momentum: float) -> None:
        new_bbox = np.array(detection["bbox"], dtype=float)
        new_velocity = new_bbox - self.bbox
        self.velocity = velocity_momentum * self.velocity + (1.0 - velocity_momentum) * new_velocity
        self.bbox = new_bbox
        self.score = float(detection["score"])
        self.category = int(detection.get("category", self.category))
        self.last_frame = frame_id
        self.hits += 1
        self.time_since_update = 0
        self.y_last = float(self.bbox[1] + self.bbox[3] / 2.0)

    def mark_missed(self) -> None:
        self.time_since_update += 1

    def to_output(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "score": round(self.score, 4),
            "category": self.category,
            "bbox": [round(float(v), 2) for v in self.bbox.tolist()],
        }


def load_frames(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return dict(data.get("meta", {})), data.get("frames", [])
    return {}, data


def run_tracker(frames: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active: list[Track] = []
    finished: list[Track] = []
    next_id = 1
    output_frames: list[dict[str, Any]] = []
    total_input_detections = 0
    total_kept_detections = 0

    for frame in frames:
        frame_id = int(frame["frame_id"])
        detections = [
            {
                "bbox": [float(v) for v in det["bbox"]],
                "score": float(det.get("score", 1.0)),
                "category": int(det.get("category", 0)),
                "source_track_id": int(det.get("track_id", -1)),
            }
            for det in frame.get("tracks", [])
            if float(det.get("score", 1.0)) >= args.low_thresh
        ]
        total_input_detections += len(frame.get("tracks", []))
        detections = nms_detections(detections, args.nms_iou)
        total_kept_detections += len(detections)

        high_indices = [i for i, det in enumerate(detections) if det["score"] >= args.track_thresh]
        low_indices = [i for i, det in enumerate(detections) if args.low_thresh <= det["score"] < args.track_thresh]
        unmatched_track_indices = list(range(len(active)))
        matched_det_indices: set[int] = set()
        updated_indices: set[int] = set()

        def match_subset(track_indices: list[int], det_indices: list[int], threshold: float) -> list[tuple[int, int]]:
            if not track_indices or not det_indices:
                return []
            track_boxes = np.array([active[i].predict_bbox(args.velocity_weight) for i in track_indices], dtype=float)
            det_boxes = np.array([detections[i]["bbox"] for i in det_indices], dtype=float)
            pairs = match_by_similarity(
                track_boxes=track_boxes,
                det_boxes=det_boxes,
                iou_threshold=threshold,
                center_distance_thresh=args.center_distance_thresh,
                center_weight=args.center_weight,
            )
            return [(track_indices[t], det_indices[d]) for t, d in pairs]

        for track_index, det_index in match_subset(unmatched_track_indices, high_indices, args.match_thresh):
            active[track_index].update(detections[det_index], frame_id, args.velocity_momentum)
            updated_indices.add(track_index)
            matched_det_indices.add(det_index)

        unmatched_track_indices = [i for i in unmatched_track_indices if i not in updated_indices]
        remaining_low = [i for i in low_indices if i not in matched_det_indices]
        for track_index, det_index in match_subset(unmatched_track_indices, remaining_low, args.low_match_thresh):
            active[track_index].update(detections[det_index], frame_id, args.velocity_momentum)
            updated_indices.add(track_index)
            matched_det_indices.add(det_index)

        for track_index, track in enumerate(active):
            if track_index not in updated_indices:
                track.mark_missed()

        for det_index in high_indices:
            if det_index in matched_det_indices:
                continue
            det = detections[det_index]
            if det["score"] < args.new_track_thresh:
                continue
            track = Track(
                track_id=next_id,
                bbox=np.array(det["bbox"], dtype=float),
                score=det["score"],
                category=det["category"],
                start_frame=frame_id,
                last_frame=frame_id,
            )
            next_id += 1
            active.append(track)
            updated_indices.add(len(active) - 1)

        frame_tracks = [
            track.to_output()
            for track in active
            if track.time_since_update == 0 and track.hits >= args.min_output_hits
        ]
        output_frames.append({"frame_id": frame_id, "file_name": frame["file_name"], "tracks": frame_tracks})

        still_active: list[Track] = []
        for track in active:
            if track.time_since_update <= args.max_age:
                still_active.append(track)
            else:
                finished.append(track)
        active = still_active

    finished.extend(active)
    summary = summarize_tracks(output_frames, finished, total_input_detections, total_kept_detections, args)
    return output_frames, summary


def summarize_tracks(
    output_frames: list[dict[str, Any]],
    tracks: list[Track],
    total_input_detections: int,
    total_kept_detections: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_lengths: dict[int, int] = {}
    first_frame: dict[int, int] = {}
    last_frame: dict[int, int] = {}
    y_first: dict[int, float] = {}
    y_last: dict[int, float] = {}
    detections_per_frame: list[int] = []

    for frame in output_frames:
        frame_id = int(frame["frame_id"])
        detections_per_frame.append(len(frame["tracks"]))
        for det in frame["tracks"]:
            track_id = int(det["track_id"])
            x, y, w, h = [float(v) for v in det["bbox"]]
            output_lengths[track_id] = output_lengths.get(track_id, 0) + 1
            first_frame.setdefault(track_id, frame_id)
            last_frame[track_id] = frame_id
            y_first.setdefault(track_id, y + h / 2.0)
            y_last[track_id] = y + h / 2.0

    lengths = list(output_lengths.values())
    upward_tracks = [
        tid
        for tid, length in output_lengths.items()
        if length >= args.min_long_track_length and (y_first[tid] - y_last[tid]) >= args.min_vertical_delta
    ]
    bottom_to_top_tracks = [
        tid
        for tid, length in output_lengths.items()
        if length >= args.min_long_track_length and y_first[tid] >= args.bottom_y and y_last[tid] <= args.top_y
    ]
    return {
        "num_frames": len(output_frames),
        "input_detections": total_input_detections,
        "kept_detections_after_score_nms": total_kept_detections,
        "output_detections": sum(detections_per_frame),
        "unique_track_ids": len(output_lengths),
        "unique_ids_per_detection": round(len(output_lengths) / sum(detections_per_frame), 6)
        if sum(detections_per_frame)
        else 0.0,
        "detections_per_frame_mean": round(statistics.fmean(detections_per_frame), 6) if detections_per_frame else 0.0,
        "track_length_mean": round(statistics.fmean(lengths), 6) if lengths else 0.0,
        "track_length_median": float(statistics.median(lengths)) if lengths else 0.0,
        "track_length_max": max(lengths, default=0),
        "num_tracks_ge_30": sum(1 for length in lengths if length >= 30),
        "num_tracks_ge_60": sum(1 for length in lengths if length >= 60),
        "num_tracks_ge_120": sum(1 for length in lengths if length >= 120),
        "long_upward_tracks": len(upward_tracks),
        "bottom_to_top_tracks": len(bottom_to_top_tracks),
        "sample_long_upward_track_ids": upward_tracks[:20],
        "sample_bottom_to_top_track_ids": bottom_to_top_tracks[:20],
        "params": {
            "track_thresh": args.track_thresh,
            "low_thresh": args.low_thresh,
            "new_track_thresh": args.new_track_thresh,
            "match_thresh": args.match_thresh,
            "low_match_thresh": args.low_match_thresh,
            "max_age": args.max_age,
            "nms_iou": args.nms_iou,
            "velocity_weight": args.velocity_weight,
            "velocity_momentum": args.velocity_momentum,
            "center_distance_thresh": args.center_distance_thresh,
            "center_weight": args.center_weight,
            "min_output_hits": args.min_output_hits,
        },
    }


def write_outputs(meta: dict[str, Any], frames: list[dict[str, Any]], summary: dict[str, Any], args: argparse.Namespace) -> None:
    result = {"meta": {**meta, "postprocess": "bytetrack_iou", "summary": summary}, "frames": frames}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_mot:
        args.output_mot.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for frame in frames:
            for track in frame["tracks"]:
                x, y, w, h = [float(v) for v in track["bbox"]]
                lines.append(
                    f'{frame["frame_id"]},{track["track_id"]},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{float(track["score"]):.4f},-1,-1,-1'
                )
        args.output_mot.write_text("\n".join(lines) + ("\n" if lines else ""))
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-mot", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--track-thresh", type=float, default=0.75)
    parser.add_argument("--low-thresh", type=float, default=0.2)
    parser.add_argument("--new-track-thresh", type=float, default=0.8)
    parser.add_argument("--match-thresh", type=float, default=0.2)
    parser.add_argument("--low-match-thresh", type=float, default=0.1)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument("--velocity-momentum", type=float, default=0.8)
    parser.add_argument(
        "--center-distance-thresh",
        type=float,
        default=0.0,
        help="Also allow matches whose center distance is within this many pixels. 0 disables distance matching.",
    )
    parser.add_argument("--center-weight", type=float, default=1.0, help="Weight for center-distance similarity.")
    parser.add_argument("--min-output-hits", type=int, default=1)
    parser.add_argument("--min-long-track-length", type=int, default=30)
    parser.add_argument("--min-vertical-delta", type=float, default=250.0)
    parser.add_argument("--bottom-y", type=float, default=450.0)
    parser.add_argument("--top-y", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta, frames = load_frames(args.input_json)
    output_frames, summary = run_tracker(frames, args)
    write_outputs(meta, output_frames, summary, args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[retrack] JSON -> {args.output_json}")
    if args.output_mot:
        print(f"[retrack] MOT -> {args.output_mot}")
    if args.summary_json:
        print(f"[retrack] summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
