import argparse

from tools.retrack_detections import run_tracker


def _args(**overrides):
    values = {
        "track_thresh": 0.8,
        "low_thresh": 0.1,
        "new_track_thresh": 0.8,
        "match_thresh": 0.1,
        "low_match_thresh": 0.1,
        "max_age": 2,
        "nms_iou": 1.0,
        "velocity_weight": 1.0,
        "velocity_momentum": 0.0,
        "min_output_hits": 1,
        "min_long_track_length": 2,
        "min_vertical_delta": 10.0,
        "bottom_y": 50.0,
        "top_y": 30.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_retrack_keeps_one_id_when_source_ids_change():
    frames = [
        {"frame_id": 1, "file_name": "000001.jpg", "tracks": [{"track_id": 10, "score": 0.95, "category": 0, "bbox": [10, 60, 20, 20]}]},
        {"frame_id": 2, "file_name": "000002.jpg", "tracks": [{"track_id": 11, "score": 0.96, "category": 0, "bbox": [11, 50, 20, 20]}]},
        {"frame_id": 3, "file_name": "000003.jpg", "tracks": [{"track_id": 12, "score": 0.97, "category": 0, "bbox": [12, 40, 20, 20]}]},
    ]

    output, summary = run_tracker(frames, _args())

    ids = [frame["tracks"][0]["track_id"] for frame in output]
    assert ids == [1, 1, 1]
    assert summary["unique_track_ids"] == 1
    assert summary["track_length_max"] == 3


def test_retrack_uses_low_score_detection_for_existing_track_but_not_new_track():
    frames = [
        {"frame_id": 1, "file_name": "000001.jpg", "tracks": [{"track_id": 10, "score": 0.95, "category": 0, "bbox": [10, 60, 20, 20]}]},
        {"frame_id": 2, "file_name": "000002.jpg", "tracks": [{"track_id": 11, "score": 0.30, "category": 0, "bbox": [11, 50, 20, 20]}]},
        {"frame_id": 3, "file_name": "000003.jpg", "tracks": [{"track_id": 12, "score": 0.30, "category": 0, "bbox": [90, 50, 20, 20]}]},
    ]

    output, summary = run_tracker(frames, _args())

    assert [track["track_id"] for track in output[0]["tracks"]] == [1]
    assert [track["track_id"] for track in output[1]["tracks"]] == [1]
    assert output[2]["tracks"] == []
    assert summary["unique_track_ids"] == 1
