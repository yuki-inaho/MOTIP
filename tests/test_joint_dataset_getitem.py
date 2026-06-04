"""Characterization regression test for the sampling path (Bug2).

Bug2 was *no coverage* of the path
    ``JointDataset.__getitem__`` -> ``build_transforms`` -> ``GenerateIDLabels``
        / ``TurnIntoTrajectoryAndUnknown``
which produces the per-frame ``trajectory_*`` / ``unknown_*`` ID annotations
that the training loop (and ``collate_fn``) depend on.

This test pins the current (correct) behaviour so future refactors cannot
silently drop those keys. It is expected to PASS on the current code: it closes
a coverage gap rather than fixing a defect.
"""

import json
from pathlib import Path

import torch
from PIL import Image

from tools.convert_coco_tracklets_to_pseudomot import convert_coco_tracklets_to_pseudomot


# Smoke-config-equivalent transform settings (see
# configs/train_tracklet_pseudomot_smoke.yaml). Kept tiny for a fast, CPU-only,
# deterministic test.
_SMOKE_TRANSFORM_CONFIG = {
    "AUG_MAX_SHIFT_RATIO": 0.0,
    "AUG_OVERFLOW_BBOX": False,
    "AUG_RESIZE_SCALES": [320],
    "AUG_MAX_SIZE": 640,
    "AUG_RANDOM_RESIZE": [320],
    "AUG_RANDOM_CROP_MIN": 256,
    "AUG_RANDOM_CROP_MAX": 320,
    "AUG_BRIGHTNESS": 0.0,
    "AUG_CONTRAST": 0.0,
    "AUG_SATURATION": 0.0,
    "AUG_HUE": 0.0,
    "AUG_COLOR_JITTER_V2": False,
    "AUG_NUM_GROUPS": 1,
    "AUG_TRAJECTORY_OCCLUSION_PROB": 0.0,
    "AUG_TRAJECTORY_SWITCH_PROB": 0.0,
    "NUM_ID_VOCABULARY": 64,
    "NUM_TRAINING_IDS": 64,
}

_SAMPLE_LENGTH = 2  # SAMPLE_LENGTHS: [2] in the smoke config.

# The 8 ID-annotation keys that build_transforms must attach to every frame.
_TRAJECTORY_KEYS = (
    "trajectory_id_labels",
    "trajectory_id_masks",
    "trajectory_ann_idxs",
    "trajectory_times",
    "unknown_id_labels",
    "unknown_id_masks",
    "unknown_ann_idxs",
    "unknown_times",
)


def _write_image(path: Path, size: tuple[int, int] = (80, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(90, 80, 70)).save(path)


def _build_dataset(root: Path) -> Path:
    """Create a tiny 2-frame PseudoMOT dataset (mirrors test_pseudo_mot_dataset)."""
    image_dir = root / "images"
    for name in ["000001.jpg", "000002.jpg"]:
        _write_image(image_dir / name)
    coco_path = root / "tracklets.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "000001.jpg", "width": 80, "height": 60},
                    {"id": 2, "file_name": "000002.jpg", "width": 80, "height": 60},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 11], "attributes": {"track_id": 7, "score": 0.9}},
                    {"id": 2, "image_id": 2, "category_id": 1, "bbox": [2, 3, 10, 11], "attributes": {"track_id": 7, "score": 0.8}},
                ],
            }
        ),
        encoding="utf-8",
    )
    output_root = root / "datasets" / "TomatoTrackletMOT"
    convert_coco_tracklets_to_pseudomot(coco_path, image_dir, output_root, "mini_seq", link_images=False)
    return output_root


def test_getitem_produces_trajectory_id_labels(tmp_path: Path) -> None:
    from data.joint_dataset import JointDataset
    from data.transforms import build_transforms

    output_root = _build_dataset(tmp_path)
    transforms = build_transforms(_SMOKE_TRANSFORM_CONFIG)

    joint = JointDataset(
        data_root=str(output_root.parent),
        datasets=["PseudoMOT"],
        splits=["train"],
        transforms=transforms,
        pseudomot_sub_dir="TomatoTrackletMOT",
    )
    joint.set_sample_details(sample_length=_SAMPLE_LENGTH, sample_interval=1)
    assert len(joint) == 1, "exactly one legal 2-frame clip is expected"

    # __getitem__ consumes a sample-spec dict (see data/naive_sampler.py), not an int.
    dataset, split, sequence, begin = joint.sample_begins[0]
    info = {
        "dataset": dataset,
        "split": split,
        "sequence": sequence,
        "frame_idxs": list(range(begin, begin + _SAMPLE_LENGTH)),
    }
    images, annotations, metas = joint[info]

    # Return-structure contract.
    assert len(annotations) == _SAMPLE_LENGTH
    assert len(metas) == _SAMPLE_LENGTH

    # Every frame must carry all 8 trajectory/unknown ID-annotation keys ...
    for frame_ann in annotations:
        for key in _TRAJECTORY_KEYS:
            assert key in frame_ann, f"missing ID-annotation key '{key}' (Bug2 coverage gap)"

    # ... with a consistent (G, 1, N) shape (G = AUG_NUM_GROUPS, time-slice = 1).
    g = _SMOKE_TRANSFORM_CONFIG["AUG_NUM_GROUPS"]
    ref_shape = annotations[0]["trajectory_id_labels"].shape
    assert ref_shape[0] == g
    assert ref_shape[1] == 1
    n = ref_shape[2]
    assert n >= 1, "the single tracked object must yield at least one ID slot"
    for frame_ann in annotations:
        for key in _TRAJECTORY_KEYS:
            assert frame_ann[key].shape == (g, 1, n), (
                f"key '{key}' has shape {tuple(frame_ann[key].shape)}, expected {(g, 1, n)}"
            )

    # dtype sanity: labels/idxs are int64, masks are bool.
    assert annotations[0]["trajectory_id_labels"].dtype == torch.int64
    assert annotations[0]["trajectory_id_masks"].dtype == torch.bool
