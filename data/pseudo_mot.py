# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
from collections import defaultdict
from configparser import ConfigParser

import torch

from .one_dataset import OneDataset
from .util import append_annotation, is_legal


class PseudoMOT(OneDataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "TomatoTrackletMOT",
            split: str = "train",
            load_annotation: bool = True,
            label_file_name: str = "gt.txt",
            allow_empty_frames: bool = False,
    ):
        self.label_file_name = label_file_name
        self.allow_empty_frames = allow_empty_frames
        super().__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )

        self.sequence_infos = self._get_sequence_infos()
        self.image_paths = self._get_image_paths()
        if self.load_annotation:
            self.annotations = self._get_annotations()
        return

    def _get_sequence_names(self):
        split_dir = os.path.join(self.data_dir, self.split)
        return sorted(os.listdir(split_dir)) if os.path.isdir(split_dir) else []

    def _get_sequence_infos(self):
        sequence_infos = {}
        for sequence_name in self._get_sequence_names():
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            parser = ConfigParser()
            parser.optionxform = str
            parser.read(os.path.join(sequence_dir, "seqinfo.ini"))
            sequence_infos[sequence_name] = {
                "width": int(parser["Sequence"]["imWidth"]),
                "height": int(parser["Sequence"]["imHeight"]),
                "length": int(parser["Sequence"]["seqLength"]),
                "is_static": False,
            }
        return sequence_infos

    def _get_image_paths(self):
        image_paths = defaultdict(list)
        for sequence_name in self._get_sequence_names():
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            for frame_idx in range(self.sequence_infos[sequence_name]["length"]):
                image_paths[sequence_name].append(self._get_image_path(sequence_dir, frame_idx))
        return image_paths

    @staticmethod
    def _get_sequence_dir(data_dir, split, sequence_name):
        return str(os.path.join(data_dir, split, sequence_name))

    @staticmethod
    def _get_image_path(sequence_dir, frame_idx):
        return str(os.path.join(sequence_dir, "img1", f"{frame_idx + 1:08d}.jpg"))

    def _init_annotations(self, sequence_names):
        annotations = {}
        for sequence_name in sequence_names:
            annotations[sequence_name] = []
            for _ in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name].append(
                    {
                        "id": torch.zeros((0,), dtype=torch.int64),
                        "category": torch.zeros((0,), dtype=torch.int64),
                        "bbox": torch.zeros((0, 4), dtype=torch.float32),
                        "visibility": torch.zeros((0,), dtype=torch.float32),
                    }
                )
        return annotations

    def _get_annotations(self):
        sequence_names = self._get_sequence_names()
        annotations = self._init_annotations(sequence_names)
        for sequence_name in sequence_names:
            gt_path = os.path.join(
                self._get_sequence_dir(self.data_dir, self.split, sequence_name),
                "gt",
                self.label_file_name,
            )
            with open(gt_path, encoding="utf-8") as gt_file:
                for line in gt_file:
                    if not line.strip():
                        continue
                    values = line.strip().split(",")
                    if len(values) != 9:
                        raise ValueError(f"PseudoMOT label line must have 9 columns: {line}")
                    frame_id, obj_id, x, y, w, h, _score, category, visibility = values
                    frame_idx = int(frame_id) - 1
                    if frame_idx < 0 or frame_idx >= self.sequence_infos[sequence_name]["length"]:
                        raise ValueError(f"frame_id out of range in {gt_path}: {frame_id}")
                    annotations[sequence_name][frame_idx] = append_annotation(
                        annotation=annotations[sequence_name][frame_idx],
                        obj_id=int(obj_id),
                        category=max(int(category) - 1, 0),
                        bbox=[float(x), float(y), float(w), float(h)],
                        visibility=float(visibility),
                    )

        for sequence_name in sequence_names:
            for frame_idx in range(self.sequence_infos[sequence_name]["length"]):
                legal = is_legal(annotations[sequence_name][frame_idx])
                annotations[sequence_name][frame_idx]["is_legal"] = bool(legal or self.allow_empty_frames)
        return annotations
