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


def _build_dataset_with_empty_frame(root: Path) -> Path:
    """Two frames, but only frame 1 has an annotation -> frame 2 is empty."""
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
                ],
            }
        ),
        encoding="utf-8",
    )
    output_root = root / "datasets" / "TomatoTrackletMOT"
    convert_coco_tracklets_to_pseudomot(coco_path, image_dir, output_root, "mini_seq", link_images=False)
    return output_root


def test_get_sequence_names_ignores_non_directory_entries(tmp_path: Path) -> None:
    from data.pseudo_mot import PseudoMOT

    output_root = _build_dataset(tmp_path)
    # Drop stray non-directory files into the split dir; they must be ignored.
    split_dir = output_root / "train"
    (split_dir / "README.txt").write_text("not a sequence", encoding="utf-8")
    (split_dir / ".DS_Store").write_text("", encoding="utf-8")

    dataset = PseudoMOT(data_root=str(output_root.parent), sub_dir="TomatoTrackletMOT", split="train")

    # Only the real sequence directory is enumerated (stray files ignored).
    assert dataset._get_sequence_names() == ["mini_seq"]
    assert list(dataset.get_sequence_infos().keys()) == ["mini_seq"]


def test_get_sequence_names_preserves_sorted_order(tmp_path: Path) -> None:
    """Regression: multiple sequence dirs must come back in sorted order, and a
    stray non-dir file must not perturb that order."""
    from data.pseudo_mot import PseudoMOT

    output_root = _build_dataset(tmp_path)  # creates "mini_seq"
    split_dir = output_root / "train"
    # Clone the mini_seq layout under two new names so sorted != insertion order.
    src = split_dir / "mini_seq"
    for extra in ["b_seq", "a_seq"]:
        dst = split_dir / extra
        dst.mkdir()
        for sub in ["img1", "gt"]:
            (dst / sub).mkdir()
            for f in (src / sub).iterdir():
                (dst / sub / f.name).write_bytes(f.read_bytes())
        (dst / "seqinfo.ini").write_bytes((src / "seqinfo.ini").read_bytes())
    (split_dir / "stray.txt").write_text("x", encoding="utf-8")

    dataset = PseudoMOT(data_root=str(output_root.parent), sub_dir="TomatoTrackletMOT", split="train")
    assert dataset._get_sequence_names() == ["a_seq", "b_seq", "mini_seq"]


def test_allow_empty_frames_controls_is_legal(tmp_path: Path) -> None:
    from data.pseudo_mot import PseudoMOT

    output_root = _build_dataset_with_empty_frame(tmp_path)

    # Default: empty frame is NOT legal.
    strict = PseudoMOT(
        data_root=str(output_root.parent), sub_dir="TomatoTrackletMOT", split="train",
        allow_empty_frames=False,
    )
    strict_anns = strict.get_annotations()["mini_seq"]
    assert strict_anns[0]["is_legal"] is True       # frame 0 has an annotation
    assert strict_anns[1]["is_legal"] is False      # frame 1 is empty

    # allow_empty_frames=True legalizes the empty frame.
    lenient = PseudoMOT(
        data_root=str(output_root.parent), sub_dir="TomatoTrackletMOT", split="train",
        allow_empty_frames=True,
    )
    lenient_anns = lenient.get_annotations()["mini_seq"]
    assert lenient_anns[0]["is_legal"] is True
    assert lenient_anns[1]["is_legal"] is True       # empty frame now legal


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
