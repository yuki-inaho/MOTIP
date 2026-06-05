"""Run MOTIP tracking inference over an image sequence and dump results.

Output:
  - JSON: per-frame tracks ``{frame_id, file_name, tracks:[{track_id, score, category, bbox:[x,y,w,h]}]}``
  - (optional) MOTChallenge txt: ``frame,id,x,y,w,h,score,-1,-1,-1``

The model is built from the training config (so ``NUM_ID_VOCABULARY`` etc. match the
checkpoint) and the online MOTIP RuntimeTracker is run frame-by-frame. bbox is in the
ORIGINAL image pixel space ([x, y, w, h], top-left + size).

No implicit fallback: missing config / checkpoint / image dir, or absent CUDA, raise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision.transforms import functional as F

from configs.util import load_super_config
from models.misc import load_checkpoint
from models.motip import build as build_model
from models.runtime_tracker import RuntimeTracker
from utils.misc import yaml_to_dict
from utils.nested_tensor import nested_tensor_from_tensor_list

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_frames(image_dir: Path) -> list[Path]:
    frames = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not frames:
        raise FileNotFoundError(f"No image frames ({sorted(IMAGE_EXTS)}) under {image_dir}")
    return frames


def preprocess(frame_bgr, max_shorter: int, max_longer: int, dtype: torch.dtype, device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = F.to_tensor(rgb)
    image = F.resize(image, size=max_shorter, max_size=max_longer)
    image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if dtype != torch.float32:
        image = image.to(dtype)
    return image.to(device)


def build_inference_model(config: dict, checkpoint: Path, use_ema: bool, dtype: torch.dtype, device):
    model, _ = build_model(config)
    if use_ema:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if "ema" not in state:
            raise ValueError(f"--use-ema requested but checkpoint has no 'ema' weights: {checkpoint}")
        model.load_state_dict(state["ema"]["module"])
    else:
        load_checkpoint(model, str(checkpoint))
    model.eval().to(device)
    if dtype == torch.float16:
        model.half()
    return model


def run_inference(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MOTIP inference.")
    device = torch.device("cuda")

    config_path, checkpoint, image_dir = Path(args.config), Path(args.checkpoint), Path(args.image_dir)
    for path in (config_path, checkpoint, image_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    config = yaml_to_dict(str(config_path))
    if config.get("SUPER_CONFIG_PATH"):
        config = load_super_config(config, config["SUPER_CONFIG_PATH"])

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = build_inference_model(config, checkpoint, args.use_ema, dtype, device)

    frames = list_frames(image_dir)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    height, width = first.shape[:2]

    def cfg(name, cli, default):
        return cli if cli is not None else type(default)(config.get(name, default))

    det_thresh = cfg("DET_THRESH", args.det_thresh, 0.3)
    newborn_thresh = cfg("NEWBORN_THRESH", args.newborn_thresh, 0.6)
    id_thresh = cfg("ID_THRESH", args.id_thresh, 0.2)
    miss_tolerance = cfg("MISS_TOLERANCE", args.miss_tolerance, 30)
    assignment = args.assignment_protocol or str(config.get("ASSIGNMENT_PROTOCOL", "object-max"))
    area_thresh = int(config.get("AREA_THRESH", 0))

    tracker = RuntimeTracker(
        model=model,
        sequence_hw=(height, width),
        assignment_protocol=assignment,
        miss_tolerance=miss_tolerance,
        det_thresh=det_thresh,
        newborn_thresh=newborn_thresh,
        id_thresh=id_thresh,
        area_thresh=area_thresh,
        only_detr=args.only_detr,
        dtype=dtype,
    )

    frames_out: list[dict] = []
    mot_lines: list[str] = []
    with torch.inference_mode():
        for frame_id, frame_path in enumerate(frames, start=1):
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            tensor = preprocess(bgr, args.max_shorter, args.max_longer, dtype, device)
            tracker.update(nested_tensor_from_tensor_list([tensor]))
            results = tracker.get_track_results()
            ids = results.get("id")
            boxes, scores, cats = results.get("bbox"), results.get("score"), results.get("category")
            tracks: list[dict] = []
            for i in range(0 if ids is None else len(ids)):
                x, y, w, h = (float(v) for v in boxes[i].detach().cpu().tolist())
                track_id, score, category = int(ids[i]), float(scores[i]), int(cats[i])
                tracks.append({
                    "track_id": track_id,
                    "score": round(score, 4),
                    "category": category,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                })
                mot_lines.append(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1")
            frames_out.append({"frame_id": frame_id, "file_name": frame_path.name, "tracks": tracks})
            if frame_id % 100 == 0 or frame_id == len(frames):
                print(f"[infer] {frame_id}/{len(frames)} frames processed")

    unique_ids = sorted({t["track_id"] for f in frames_out for t in f["tracks"]})
    result = {
        "meta": {
            "config": str(config_path),
            "checkpoint": str(checkpoint),
            "use_ema": args.use_ema,
            "image_dir": str(image_dir),
            "num_frames": len(frames),
            "image_hw": [height, width],
            "num_id_vocabulary": int(config.get("NUM_ID_VOCABULARY")),
            "thresholds": {
                "det": det_thresh, "newborn": newborn_thresh, "id": id_thresh,
                "miss_tolerance": miss_tolerance, "assignment": assignment, "area": area_thresh,
                "only_detr": bool(args.only_detr),
            },
            "num_unique_track_ids": len(unique_ids),
            "num_detections": len(mot_lines),
        },
        "frames": frames_out,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[infer] JSON -> {out_json} "
          f"({len(frames_out)} frames, {len(unique_ids)} unique ids, {len(mot_lines)} detections)")
    if args.output_mot:
        out_mot = Path(args.output_mot)
        out_mot.parent.mkdir(parents=True, exist_ok=True)
        out_mot.write_text("\n".join(mot_lines) + ("\n" if mot_lines else ""), encoding="utf-8")
        print(f"[infer] MOTChallenge txt -> {out_mot} ({len(mot_lines)} rows)")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOTIP tracking inference -> JSON (+ MOTChallenge txt).")
    parser.add_argument("--config", default="./configs/train_tracklet_pseudomot_full.yaml")
    parser.add_argument("--checkpoint", default="./outputs/tracklet_pseudomot_full/checkpoint_7.pth")
    parser.add_argument("--image-dir", default="./datasets/TomatoTrackletMOT/train/nyx660_jun04/img1")
    parser.add_argument("--output-json", default="./outputs/tracklet_pseudomot_full/infer/tracks.json")
    parser.add_argument("--output-mot", default="./outputs/tracklet_pseudomot_full/infer/tracks_mot.txt")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights stored in the checkpoint.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames.")
    parser.add_argument("--max-shorter", type=int, default=600)
    parser.add_argument("--max-longer", type=int, default=1440)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--det-thresh", type=float, default=None)
    parser.add_argument("--newborn-thresh", type=float, default=None)
    parser.add_argument("--id-thresh", type=float, default=None)
    parser.add_argument("--miss-tolerance", type=int, default=None)
    parser.add_argument("--assignment-protocol", type=str, default=None)
    parser.add_argument(
        "--only-detr",
        action="store_true",
        help="Bypass the MOTIP ID decoder and dump DETR detections only before external re-tracking.",
    )
    return parser.parse_args()


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
