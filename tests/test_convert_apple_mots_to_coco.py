import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.convert_apple_mots_to_coco import convert_apple_mots_to_coco


def _write_rgb(path: Path, size: tuple[int, int] = (30, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def _write_mask(path: Path, values: list[tuple[int, tuple[slice, slice]]], size: tuple[int, int] = (20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(size, dtype=np.uint16)
    for encoded_id, region in values:
        mask[region] = encoded_id
    Image.fromarray(mask).save(path)


def test_convert_apple_mots_to_coco_writes_coco_annotations(tmp_path: Path) -> None:
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
    _write_mask(root / "train" / "instances" / "0000" / "000001.png", [(1000, (slice(3, 7), slice(2, 5)))])

    summary = convert_apple_mots_to_coco(
        applemots_root=root,
        output_root=tmp_path / "datasets" / "AppleMOTSCOCO",
        splits=["train"],
        link_images=False,
    )

    output = tmp_path / "datasets" / "AppleMOTSCOCO"
    coco = json.loads((output / "annotations" / "applemots_train.json").read_text())
    persisted = json.loads((output / "conversion_summary.json").read_text())

    assert coco["categories"] == [{"id": 1, "name": "apple"}]
    assert [image["file_name"] for image in coco["images"]] == [
        "train/0000/000000.png",
        "train/0000/000001.png",
    ]
    assert [annotation["bbox"] for annotation in coco["annotations"]] == [
        [1.0, 2.0, 3.0, 4.0],
        [7.0, 10.0, 2.0, 3.0],
        [2.0, 3.0, 3.0, 4.0],
    ]
    assert [annotation["area"] for annotation in coco["annotations"]] == [12.0, 6.0, 12.0]
    assert [annotation["attributes"]["track_id"] for annotation in coco["annotations"]] == [1, 2, 1]
    assert summary["by_split"]["train"]["num_frames"] == 2
    assert summary["by_split"]["train"]["num_objects"] == 3
    assert persisted["by_split"]["train"]["num_tracks"] == 2
    assert (output / "images" / "train" / "0000" / "000000.png").is_file()


def test_convert_apple_mots_to_coco_symlinks_and_reports_extra_instances(tmp_path: Path) -> None:
    root = tmp_path / "APPLE_MOTS"
    _write_rgb(root / "testing" / "images" / "0010" / "000000.png")
    _write_mask(root / "testing" / "instances" / "0010" / "000000.png", [(1001, (slice(1, 4), slice(2, 5)))])
    _write_mask(root / "testing" / "instances" / "0010" / "000096.png", [(1002, (slice(1, 2), slice(1, 2)))])

    summary = convert_apple_mots_to_coco(
        applemots_root=root,
        output_root=tmp_path / "datasets" / "AppleMOTSCOCO",
        splits=["testing"],
        link_images=True,
    )

    linked = tmp_path / "datasets" / "AppleMOTSCOCO" / "images" / "testing" / "0010" / "000000.png"
    assert linked.is_symlink()
    assert linked.exists()
    assert linked.resolve() == (root / "testing" / "images" / "0010" / "000000.png").resolve()
    assert summary["sequences"][0]["num_extra_instance_files"] == 1
    assert summary["sequences"][0]["extra_instance_files"] == ["000096"]


def test_convert_apple_mots_to_coco_can_write_zero_based_category_for_deim(tmp_path: Path) -> None:
    root = tmp_path / "APPLE_MOTS"
    _write_rgb(root / "train" / "images" / "0000" / "000000.png")
    _write_mask(root / "train" / "instances" / "0000" / "000000.png", [(1000, (slice(1, 3), slice(2, 4)))])

    convert_apple_mots_to_coco(
        applemots_root=root,
        output_root=tmp_path / "datasets" / "AppleMOTSCOCO_DEIM",
        splits=["train"],
        link_images=False,
        category_id=0,
    )

    coco = json.loads((tmp_path / "datasets" / "AppleMOTSCOCO_DEIM" / "annotations" / "applemots_train.json").read_text())
    summary = json.loads((tmp_path / "datasets" / "AppleMOTSCOCO_DEIM" / "conversion_summary.json").read_text())

    assert coco["categories"] == [{"id": 0, "name": "apple"}]
    assert coco["annotations"][0]["category_id"] == 0
    assert summary["category_id"] == 0
