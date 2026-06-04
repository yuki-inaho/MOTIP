"""Tests for ``collate_fn``'s precondition handling (Bug3).

Bug3: ``data/util.py:collate_fn`` assumes every frame annotation already carries
``trajectory_id_labels`` (it indexes it directly to compute ``max_N``). If a
transform path ever omits that key, the failure surfaces as an *opaque* bare
``KeyError`` deep inside a comprehension, with no hint of the real precondition.

This pins two contracts:
  (a) normal samples (produced via the real getitem path) collate successfully
      and the trajectory keys are padded to a common ``max_N``;
  (b) a frame missing ``trajectory_id_labels`` fails with an *explicit*,
      descriptive error (no silent fallback, no opaque KeyError).
"""

from pathlib import Path

import pytest
import torch

from data.util import collate_fn
from tests.test_joint_dataset_getitem import (
    _SAMPLE_LENGTH,
    _SMOKE_TRANSFORM_CONFIG,
    _TRAJECTORY_KEYS,
    _build_dataset,
)


def _make_sample(tmp_path: Path):
    """Produce one real (images, annotations, metas) sample via the getitem path."""
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
    dataset, split, sequence, begin = joint.sample_begins[0]
    info = {
        "dataset": dataset,
        "split": split,
        "sequence": sequence,
        "frame_idxs": list(range(begin, begin + _SAMPLE_LENGTH)),
    }
    return joint[info]


def test_collate_fn_succeeds_on_normal_samples(tmp_path: Path) -> None:
    sample = _make_sample(tmp_path)
    # A batch of two identical clips exercises the padding-to-max_N path.
    batch = [sample, sample]
    out = collate_fn(batch)

    assert set(out.keys()) == {"images", "annotations", "metas"}
    assert len(out["annotations"]) == 2

    # After collation every frame's trajectory keys share the same last dim (max_N).
    max_n = max(
        ann[0]["trajectory_id_labels"].shape[-1] for ann in out["annotations"]
    )
    for clip in out["annotations"]:
        for frame_ann in clip:
            for key in _TRAJECTORY_KEYS:
                assert frame_ann[key].shape[-1] == max_n, (
                    f"key '{key}' not padded to max_N={max_n}"
                )


def test_collate_fn_raises_explicit_error_when_trajectory_key_missing(tmp_path: Path) -> None:
    sample = _make_sample(tmp_path)
    images, annotations, metas = sample

    # Drop the precondition key from the first frame of the clip.
    broken_annotations = [dict(frame) for frame in annotations]
    del broken_annotations[0]["trajectory_id_labels"]
    broken_sample = (images, broken_annotations, metas)

    with pytest.raises(KeyError) as excinfo:
        collate_fn([broken_sample])

    # The message must be descriptive (names the key + frame), not an opaque
    # bare KeyError('trajectory_id_labels').
    message = str(excinfo.value)
    assert "trajectory_id_labels" in message
    assert "collate_fn" in message, "error must identify collate_fn as the failing precondition site"


def test_collate_fn_guard_message_is_actionable(tmp_path: Path) -> None:
    """The guard should point at the transform pipeline as the likely cause."""
    sample = _make_sample(tmp_path)
    images, annotations, metas = sample
    broken_annotations = [dict(frame) for frame in annotations]
    del broken_annotations[1]["trajectory_id_labels"]

    with pytest.raises(KeyError) as excinfo:
        collate_fn([(images, broken_annotations, metas)])

    assert "transform" in str(excinfo.value).lower()
    # Also sanity-check that a fully-valid batch still works (no false positives).
    assert collate_fn([sample]) is not None
    # And int tensors stay intact.
    assert torch.is_tensor(collate_fn([sample])["annotations"][0][0]["trajectory_id_labels"])
