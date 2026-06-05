import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.convert_apple_mots_to_pseudomot import convert_apple_mots_to_pseudomot


def _write_rgb(path: Path, size: tuple[int, int] = (30, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 40, 60)).save(path)


def _write_mask(path: Path, values: list[tuple[int, tuple[slice, slice]]], size: tuple[int, int] = (20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(size, dtype=np.uint16)
    for encoded_id, region in values:
        mask[region] = encoded_id
    Image.fromarray(mask).save(path)


def test_convert_apple_mots_to_pseudomot_writes_mot_files(tmp_path: Path) -> None:
    root = tmp_path / "APPLE_MOTS"
    for frame in ["000000", "000001"]:
        _write_rgb(root / "train" / "images" / "0000" / f"{frame}.png")
    _write_mask(
        root / "train" / "instances" / "0000" / "000000.png",
        [
            (1000, (slice(2, 6), slice(1, 4))),
            (1002, (slice(10, 13), slice(7, 9))),
        ],
    )
    _write_mask(
        root / "train" / "instances" / "0000" / "000001.png",
        [
            (1000, (slice(3, 7), slice(2, 5))),
        ],
    )

    summary = convert_apple_mots_to_pseudomot(
        applemots_root=root,
        output_root=tmp_path / "datasets" / "AppleMOTSPseudoMOT",
        splits=["train"],
        frame_rate=12,
        link_images=False,
    )

    seq = tmp_path / "datasets" / "AppleMOTSPseudoMOT" / "train" / "0000"
    gt_lines = (seq / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines()
    seqinfo = (seq / "seqinfo.ini").read_text(encoding="utf-8")
    persisted = json.loads((tmp_path / "datasets" / "AppleMOTSPseudoMOT" / "conversion_summary.json").read_text())

    assert gt_lines == [
        "1,1,1.000,2.000,3.000,4.000,1.0000,1,1.000",
        "1,2,7.000,10.000,2.000,3.000,1.0000,1,1.000",
        "2,1,2.000,3.000,3.000,4.000,1.0000,1,1.000",
    ]
    assert "frameRate=12" in seqinfo
    assert "seqLength=2" in seqinfo
    assert (seq / "img1" / "00000001.jpg").is_file()
    assert summary["by_split"]["train"]["num_frames"] == 2
    assert summary["by_split"]["train"]["num_objects"] == 3
    assert persisted["sequences"][0]["num_tracks"] == 2
    assert persisted["sequences"][0]["encoded_id_min"] == 1000
    assert persisted["sequences"][0]["encoded_id_max"] == 1002


def test_convert_apple_mots_to_pseudomot_symlinks_and_reports_extra_instances(tmp_path: Path) -> None:
    root = tmp_path / "APPLE_MOTS"
    _write_rgb(root / "testing" / "images" / "0010" / "000000.png")
    _write_mask(root / "testing" / "instances" / "0010" / "000000.png", [(1001, (slice(1, 4), slice(2, 5)))])
    _write_mask(root / "testing" / "instances" / "0010" / "000096.png", [(1002, (slice(1, 2), slice(1, 2)))])

    summary = convert_apple_mots_to_pseudomot(
        applemots_root=root,
        output_root=tmp_path / "datasets" / "AppleMOTSPseudoMOT",
        splits=["testing"],
        link_images=True,
    )

    linked = tmp_path / "datasets" / "AppleMOTSPseudoMOT" / "testing" / "0010" / "img1" / "00000001.jpg"
    assert linked.is_symlink()
    assert linked.exists()
    assert linked.resolve() == (root / "testing" / "images" / "0010" / "000000.png").resolve()
    assert summary["sequences"][0]["num_extra_instance_files"] == 1
    assert summary["sequences"][0]["extra_instance_files"] == ["000096"]
