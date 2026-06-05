import json
from pathlib import Path

from PIL import Image

from tools.convert_track_json_to_pseudomot import convert_track_json_to_pseudomot


def _write_image(path: Path, size: tuple[int, int] = (80, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(12, 34, 56)).save(path)


def test_convert_track_json_to_pseudomot_writes_mot_challenge_files(tmp_path: Path) -> None:
    image_dir = tmp_path / "img1"
    _write_image(image_dir / "00000001.jpg")
    _write_image(image_dir / "00000002.jpg")

    track_json = tmp_path / "tracks.json"
    track_json.write_text(
        json.dumps(
            {
                "meta": {"source": "unit"},
                "frames": [
                    {
                        "frame_id": 1,
                        "file_name": "00000001.jpg",
                        "tracks": [
                            {"track_id": 7, "score": 0.91, "category": 0, "bbox": [1.0, 2.0, 10.0, 11.0]},
                            {"track_id": 9, "score": 0.81, "category": 0, "bbox": [4.0, 5.0, 0.0, 11.0]},
                        ],
                    },
                    {
                        "frame_id": 2,
                        "file_name": "00000002.jpg",
                        "tracks": [
                            {"track_id": 7, "score": 0.87, "category": 0, "bbox": [3.0, 4.0, 12.0, 13.0]},
                            {"track_id": 11, "score": 0.66, "category": 0, "bbox": [20.0, 21.0, 5.0, 6.0]},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = convert_track_json_to_pseudomot(
        track_json=track_json,
        image_dir=image_dir,
        output_root=tmp_path / "datasets" / "TrackJSONMOT",
        sequence_name="mini_track_json",
        split="train",
        frame_rate=30,
        link_images=False,
    )

    root = tmp_path / "datasets" / "TrackJSONMOT" / "train" / "mini_track_json"
    gt_lines = (root / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines()
    seqinfo = (root / "seqinfo.ini").read_text(encoding="utf-8")
    persisted = json.loads((tmp_path / "datasets" / "TrackJSONMOT" / "conversion_summary.json").read_text(encoding="utf-8"))

    assert gt_lines == [
        "1,7,1.000,2.000,10.000,11.000,0.9100,1,1.000",
        "2,7,3.000,4.000,12.000,13.000,0.8700,1,1.000",
        "2,11,20.000,21.000,5.000,6.000,0.6600,1,1.000",
    ]
    assert "frameRate=30" in seqinfo
    assert "seqLength=2" in seqinfo
    assert summary["num_frames"] == 2
    assert summary["num_input_tracks"] == 4
    assert summary["num_skipped_degenerate"] == 1
    assert summary["num_objects"] == 3
    assert summary["num_tracks"] == 2
    assert summary["num_empty_frames"] == 0
    assert persisted["source_meta"] == {"source": "unit"}


def test_convert_track_json_symlinks_resolve_from_dataset_directory(tmp_path: Path) -> None:
    image_dir = tmp_path / "img1"
    _write_image(image_dir / "00000001.jpg")
    track_json = tmp_path / "tracks.json"
    track_json.write_text(
        json.dumps(
            [
                {
                    "frame_id": 1,
                    "file_name": "00000001.jpg",
                    "tracks": [{"track_id": 1, "score": 0.9, "category": 0, "bbox": [1.0, 2.0, 3.0, 4.0]}],
                }
            ]
        ),
        encoding="utf-8",
    )

    convert_track_json_to_pseudomot(
        track_json=track_json,
        image_dir=image_dir,
        output_root=tmp_path / "datasets" / "TrackJSONMOT",
        sequence_name="mini_track_json",
        split="train",
        link_images=True,
    )

    linked = tmp_path / "datasets" / "TrackJSONMOT" / "train" / "mini_track_json" / "img1" / "00000001.jpg"
    assert linked.is_symlink()
    assert linked.exists()
    assert linked.resolve() == (image_dir / "00000001.jpg").resolve()
