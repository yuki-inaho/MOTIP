"""Tune MOTIP bbox re-tracking parameters with Optuna.

The input is the JSON written by ``tools/infer_tracklet.py``. Source
``track_id`` values are ignored by ``tools/retrack_detections.py``; this tuner
optimizes the post-processing thresholds that turn per-frame boxes into longer
tracklets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import TPESampler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.retrack_detections import load_frames, run_tracker, write_outputs


TRACK_THRESH_CHOICES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
LOW_THRESH_CHOICES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
NEW_TRACK_THRESH_CHOICES = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
MATCH_THRESH_CHOICES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22]
MAX_AGE_CHOICES = [40, 50, 60, 75, 90, 120, 150, 180, 240, 300]
NMS_IOU_CHOICES = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
VELOCITY_WEIGHT_CHOICES = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
VELOCITY_MOMENTUM_CHOICES = [0.50, 0.65, 0.80, 0.90, 0.95]
CENTER_DISTANCE_THRESH_CHOICES = [0.0, 20.0, 35.0, 50.0, 75.0, 100.0, 140.0]
CENTER_WEIGHT_CHOICES = [0.25, 0.5, 1.0, 2.0]


def sample_params(trial: optuna.Trial) -> dict[str, Any]:
    track_thresh = trial.suggest_categorical("track_thresh", TRACK_THRESH_CHOICES)
    low_thresh = trial.suggest_categorical("low_thresh", LOW_THRESH_CHOICES)
    new_track_thresh = trial.suggest_categorical("new_track_thresh", NEW_TRACK_THRESH_CHOICES)
    match_thresh = trial.suggest_categorical("match_thresh", MATCH_THRESH_CHOICES)
    low_match_thresh = trial.suggest_categorical("low_match_thresh", MATCH_THRESH_CHOICES)
    if low_thresh > track_thresh:
        raise optuna.TrialPruned("low_thresh must not exceed track_thresh")
    if new_track_thresh < track_thresh:
        raise optuna.TrialPruned("new_track_thresh must not be lower than track_thresh")
    if low_match_thresh > match_thresh:
        raise optuna.TrialPruned("low_match_thresh must not exceed match_thresh")
    return {
        "track_thresh": track_thresh,
        "low_thresh": low_thresh,
        "new_track_thresh": new_track_thresh,
        "match_thresh": match_thresh,
        "low_match_thresh": low_match_thresh,
        "max_age": trial.suggest_categorical("max_age", MAX_AGE_CHOICES),
        "nms_iou": trial.suggest_categorical("nms_iou", NMS_IOU_CHOICES),
        "velocity_weight": trial.suggest_categorical("velocity_weight", VELOCITY_WEIGHT_CHOICES),
        "velocity_momentum": trial.suggest_categorical("velocity_momentum", VELOCITY_MOMENTUM_CHOICES),
        "center_distance_thresh": trial.suggest_categorical("center_distance_thresh", CENTER_DISTANCE_THRESH_CHOICES),
        "center_weight": trial.suggest_categorical("center_weight", CENTER_WEIGHT_CHOICES),
    }


def tracker_args(params: dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        input_json=args.input_json,
        output_json=args.best_output_json or Path("/tmp/retrack_tuning_unused.json"),
        output_mot=args.best_output_mot,
        summary_json=args.best_summary_json,
        min_output_hits=1,
        min_long_track_length=args.min_long_track_length,
        min_vertical_delta=args.min_vertical_delta,
        bottom_y=args.bottom_y,
        top_y=args.top_y,
        **params,
    )


def score_summary(summary: dict[str, Any], args: argparse.Namespace) -> float:
    input_detections = max(1, int(summary["input_detections"]))
    output_ratio = float(summary["output_detections"]) / input_detections
    unique_ratio = float(summary["unique_ids_per_detection"])

    # Reward long and vertically useful tracks, not just fewer IDs.
    score = 0.0
    score += float(summary["num_tracks_ge_120"]) * 300.0
    score += float(summary["num_tracks_ge_60"]) * 12.0
    score += float(summary["bottom_to_top_tracks"]) * 10.0
    score += float(summary["long_upward_tracks"]) * 3.0
    score += float(summary["track_length_max"]) * 5.0
    score += float(summary["track_length_mean"]) * 10.0
    score += float(summary["track_length_median"]) * 5.0

    # Penalize settings that create long tracks by dropping too many detections
    # or by merging the whole scene into too few IDs.
    score -= max(0.0, args.min_output_ratio - output_ratio) * 1200.0
    score -= max(0.0, args.min_unique_ratio - unique_ratio) * 900.0
    score -= max(0.0, unique_ratio - args.max_unique_ratio) * 120.0
    return score


def trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "summary": trial.user_attrs.get("summary"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--study-json", required=True, type=Path)
    parser.add_argument("--best-output-json", type=Path)
    parser.add_argument("--best-output-mot", type=Path)
    parser.add_argument("--best-summary-json", type=Path)
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-output-ratio", type=float, default=0.70)
    parser.add_argument("--min-unique-ratio", type=float, default=0.025)
    parser.add_argument("--max-unique-ratio", type=float, default=0.15)
    parser.add_argument("--min-long-track-length", type=int, default=30)
    parser.add_argument("--min-vertical-delta", type=float, default=250.0)
    parser.add_argument("--bottom-y", type=float, default=450.0)
    parser.add_argument("--top-y", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta, frames = load_frames(args.input_json)
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        _, summary = run_tracker(frames, tracker_args(params, args))
        value = score_summary(summary, args)
        trial.set_user_attr("summary", summary)
        return value

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=False)

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No completed Optuna trials.")
    completed.sort(key=lambda trial: float(trial.value), reverse=True)
    best = completed[0]
    best_args = tracker_args(best.params, args)
    best_frames, best_summary = run_tracker(frames, best_args)

    if args.best_output_json:
        write_outputs(meta, best_frames, best_summary, best_args)

    report = {
        "input_json": str(args.input_json),
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len(completed),
        "timeout_seconds": args.timeout,
        "seed": args.seed,
        "objective_constraints": {
            "min_output_ratio": args.min_output_ratio,
            "min_unique_ratio": args.min_unique_ratio,
            "max_unique_ratio": args.max_unique_ratio,
        },
        "best_value": best.value,
        "best_params": best.params,
        "best_summary": best_summary,
        "top_trials": [trial_record(trial) for trial in completed[:20]],
    }
    args.study_json.parent.mkdir(parents=True, exist_ok=True)
    args.study_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["best_summary"], indent=2, sort_keys=True))
    print(f"[tune] best value -> {best.value:.6f}")
    print(f"[tune] study -> {args.study_json}")
    if args.best_output_json:
        print(f"[tune] best JSON -> {args.best_output_json}")


if __name__ == "__main__":
    main()
