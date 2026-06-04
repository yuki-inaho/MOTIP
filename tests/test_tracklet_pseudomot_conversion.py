import json
from pathlib import Path

from PIL import Image

from tools.convert_coco_tracklets_to_pseudomot import convert_coco_tracklets_to_pseudomot


def _write_image(path: Path, size: tuple[int, int] = (80, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(12, 34, 56)).save(path)


def test_convert_coco_tracklets_to_pseudomot_writes_mot_challenge_files(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _write_image(image_dir / "000001.jpg")
    _write_image(image_dir / "000002.jpg")

    coco_path = tmp_path / "tracklets.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 10, "file_name": "000001.jpg", "width": 80, "height": 60},
                    {"id": 11, "file_name": "000002.jpg", "width": 80, "height": 60},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "category_id": 1,
                        "bbox": [1.0, 2.0, 10.0, 11.0],
                        "attributes": {"track_id": 42, "score": 0.91},
                    },
                    {
                        "id": 2,
                        "image_id": 11,
                        "category_id": 1,
                        "bbox": [3.0, 4.0, 12.0, 13.0],
                        "attributes": {"track_id": 42, "score": 0.87},
                    },
                    {
                        "id": 3,
                        "image_id": 11,
                        "category_id": 1,
                        "bbox": [20.0, 21.0, 5.0, 6.0],
                        "attributes": {"track_id": 99, "score": 0.66},
                    },
                ],
                "categories": [{"id": 1, "name": "tomato"}],
            }
        ),
        encoding="utf-8",
    )

    summary = convert_coco_tracklets_to_pseudomot(
        coco_path=coco_path,
        image_dir=image_dir,
        output_root=tmp_path / "datasets" / "TomatoTrackletMOT",
        sequence_name="mini_seq",
        split="train",
        link_images=False,
    )

    gt_path = tmp_path / "datasets" / "TomatoTrackletMOT" / "train" / "mini_seq" / "gt" / "gt.txt"
    seqinfo_path = tmp_path / "datasets" / "TomatoTrackletMOT" / "train" / "mini_seq" / "seqinfo.ini"
    summary_path = tmp_path / "datasets" / "TomatoTrackletMOT" / "conversion_summary.json"

    assert summary["num_frames"] == 2
    assert summary["num_objects"] == 3
    assert summary["num_tracks"] == 2
    assert gt_path.read_text(encoding="utf-8").splitlines() == [
        "1,43,1.000,2.000,10.000,11.000,0.910,1,1.000",
        "2,43,3.000,4.000,12.000,13.000,0.870,1,1.000",
        "2,100,20.000,21.000,5.000,6.000,0.660,1,1.000",
    ]
    assert "seqLength=2" in seqinfo_path.read_text(encoding="utf-8")
    assert json.loads(summary_path.read_text(encoding="utf-8"))["num_tracks"] == 2
    assert (tmp_path / "datasets" / "TomatoTrackletMOT" / "train" / "mini_seq" / "img1" / "00000001.jpg").is_file()


def _degenerate_coco(image_dir: Path) -> dict:
    """COCO with 4 annotations: 2 valid, 2 degenerate (w<=0 / h<=0)."""
    _write_image(image_dir / "000001.jpg")
    _write_image(image_dir / "000002.jpg")
    return {
        "images": [
            {"id": 10, "file_name": "000001.jpg", "width": 80, "height": 60},
            {"id": 11, "file_name": "000002.jpg", "width": 80, "height": 60},
        ],
        "annotations": [
            # valid
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [1.0, 2.0, 10.0, 11.0], "attributes": {"track_id": 1, "score": 0.9}},
            # degenerate: zero width
            {"id": 2, "image_id": 10, "category_id": 1, "bbox": [3.0, 4.0, 0.0, 5.0], "attributes": {"track_id": 2, "score": 0.8}},
            # degenerate: negative height
            {"id": 3, "image_id": 11, "category_id": 1, "bbox": [5.0, 6.0, 7.0, -1.0], "attributes": {"track_id": 3, "score": 0.7}},
            # valid
            {"id": 4, "image_id": 11, "category_id": 1, "bbox": [2.0, 3.0, 12.0, 13.0], "attributes": {"track_id": 1, "score": 0.6}},
        ],
        "categories": [{"id": 1, "name": "tomato"}],
    }


def test_degenerate_bboxes_are_counted_not_silently_dropped(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    coco_path = tmp_path / "tracklets.json"
    coco_path.write_text(json.dumps(_degenerate_coco(image_dir)), encoding="utf-8")

    summary = convert_coco_tracklets_to_pseudomot(
        coco_path=coco_path,
        image_dir=image_dir,
        output_root=tmp_path / "datasets" / "TomatoTrackletMOT",
        sequence_name="mini_seq",
        split="train",
        link_images=False,
    )

    # No silent fallback: the skip must be accounted for in the summary.
    assert summary["num_input_annotations"] == 4
    assert summary["num_skipped_degenerate"] == 2
    assert summary["num_objects"] == 2
    # The accounting invariant must always hold.
    assert summary["num_input_annotations"] == summary["num_objects"] + summary["num_skipped_degenerate"]

    # The same keys/values must be persisted to conversion_summary.json.
    summary_path = tmp_path / "datasets" / "TomatoTrackletMOT" / "conversion_summary.json"
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["num_input_annotations"] == 4
    assert persisted["num_skipped_degenerate"] == 2
    assert persisted["num_objects"] == 2


def test_no_degenerate_input_keeps_num_objects_and_zero_skips(tmp_path: Path) -> None:
    """Regression: with no degenerate bboxes, num_objects is unchanged and
    num_skipped_degenerate is 0 (and the invariant still holds)."""
    image_dir = tmp_path / "images"
    _write_image(image_dir / "000001.jpg")
    _write_image(image_dir / "000002.jpg")
    coco_path = tmp_path / "tracklets.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 10, "file_name": "000001.jpg", "width": 80, "height": 60},
                    {"id": 11, "file_name": "000002.jpg", "width": 80, "height": 60},
                ],
                "annotations": [
                    {"id": 1, "image_id": 10, "category_id": 1, "bbox": [1.0, 2.0, 10.0, 11.0], "attributes": {"track_id": 42, "score": 0.91}},
                    {"id": 2, "image_id": 11, "category_id": 1, "bbox": [3.0, 4.0, 12.0, 13.0], "attributes": {"track_id": 42, "score": 0.87}},
                    {"id": 3, "image_id": 11, "category_id": 1, "bbox": [20.0, 21.0, 5.0, 6.0], "attributes": {"track_id": 99, "score": 0.66}},
                ],
                "categories": [{"id": 1, "name": "tomato"}],
            }
        ),
        encoding="utf-8",
    )

    summary = convert_coco_tracklets_to_pseudomot(
        coco_path=coco_path,
        image_dir=image_dir,
        output_root=tmp_path / "datasets" / "TomatoTrackletMOT",
        sequence_name="mini_seq",
        split="train",
        link_images=False,
    )

    # num_objects unchanged from the original (3 valid annotations -> 3 objects).
    assert summary["num_objects"] == 3
    assert summary["num_skipped_degenerate"] == 0
    assert summary["num_input_annotations"] == 3
    assert summary["num_input_annotations"] == summary["num_objects"] + summary["num_skipped_degenerate"]
