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
