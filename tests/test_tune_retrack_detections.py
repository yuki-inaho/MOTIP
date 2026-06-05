import argparse

from tools.tune_retrack_detections import score_summary


def _args(**overrides):
    values = {
        "min_output_ratio": 0.7,
        "min_unique_ratio": 0.025,
        "max_unique_ratio": 0.15,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _summary(**overrides):
    values = {
        "input_detections": 100,
        "output_detections": 80,
        "unique_ids_per_detection": 0.08,
        "num_tracks_ge_120": 0,
        "num_tracks_ge_60": 2,
        "bottom_to_top_tracks": 3,
        "long_upward_tracks": 5,
        "track_length_max": 90,
        "track_length_mean": 10.0,
        "track_length_median": 8.0,
    }
    values.update(overrides)
    return values


def test_score_prefers_longer_tracks_when_constraints_are_equal():
    short = _summary(track_length_max=90, num_tracks_ge_120=0, num_tracks_ge_60=2)
    long = _summary(track_length_max=150, num_tracks_ge_120=2, num_tracks_ge_60=4)

    assert score_summary(long, _args()) > score_summary(short, _args())


def test_score_penalizes_dropping_too_many_detections():
    good_retention = _summary(output_detections=80)
    poor_retention = _summary(output_detections=40, track_length_max=150)

    assert score_summary(good_retention, _args()) > score_summary(poor_retention, _args())
