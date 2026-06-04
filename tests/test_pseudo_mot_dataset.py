import json
from pathlib import Path

from PIL import Image

from tools.convert_coco_tracklets_to_pseudomot import convert_coco_tracklets_to_pseudomot


def _write_image(path: Path, size: tuple[int, int] = (80, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(90, 80, 70)).save(path)


def _build_dataset(root: Path) -> Path:
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


def test_pseudo_mot_loader_reads_generated_tracklet_dataset(tmp_path: Path) -> None:
    from data.joint_dataset import JointDataset
    from data.pseudo_mot import PseudoMOT

    output_root = _build_dataset(tmp_path)
    dataset = PseudoMOT(data_root=str(output_root.parent), sub_dir="TomatoTrackletMOT", split="train")
    annotations = dataset.get_annotations()["mini_seq"]

    assert dataset.get_sequence_infos()["mini_seq"]["length"] == 2
    assert dataset.get_image_paths()["mini_seq"][0].endswith("00000001.jpg")
    assert annotations[0]["id"].tolist() == [8]
    assert annotations[0]["bbox"].tolist() == [[1.0, 2.0, 10.0, 11.0]]
    assert annotations[0]["is_legal"] is True

    joint = JointDataset(
        data_root=str(output_root.parent),
        datasets=["PseudoMOT"],
        splits=["train"],
        pseudomot_sub_dir="TomatoTrackletMOT",
    )
    joint.set_sample_details(sample_length=2, sample_interval=1)

    assert joint.statistics() == ["PseudoMOT.train, 1 sequences, 2 frames."]
    assert len(joint) == 1
