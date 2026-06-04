import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm

from configs.util import load_super_config
from demo.colormap import get_color
from models.misc import load_checkpoint
from models.motip import build as build_model
from models.runtime_tracker import RuntimeTracker
from utils.misc import yaml_to_dict
from utils.nested_tensor import nested_tensor_from_tensor_list


def parse_args():
    parser = argparse.ArgumentParser("MOTIP video demo smoke runner.")
    parser.add_argument("--config", default="./configs/r50_deformable_detr_motip_dancetrack.yaml")
    parser.add_argument(
        "--checkpoint",
        default="./outputs/r50_deformable_detr_motip_dancetrack/"
        "r50_deformable_detr_motip_dancetrack.pth",
    )
    parser.add_argument("--video", default="./outputs/video_process_demo/hpop_dancers.mp4")
    parser.add_argument("--output", default="./outputs/video_process_demo/hpop_dancers_tracking_smoke.mp4")
    parser.add_argument("--summary", default="./outputs/video_process_demo/hpop_dancers_tracking_smoke.json")
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--max-shorter", type=int, default=480)
    parser.add_argument("--max-longer", type=int, default=960)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    return parser.parse_args()


def simple_transform(frame, max_shorter, max_longer, dtype):
    image = F.to_tensor(frame)
    image = F.resize(image, size=max_shorter, max_size=max_longer)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if dtype != torch.float32:
        image = image.to(dtype)
    return image.cuda()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    video_path = Path(args.video)
    for path in [config_path, checkpoint_path, video_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    config = yaml_to_dict(str(config_path))
    config = load_super_config(config, config["SUPER_CONFIG_PATH"])

    model, _ = build_model(config)
    load_checkpoint(model, str(checkpoint_path))
    model.eval().cuda()
    if dtype == torch.float16:
        model.half()

    video_cap = cv2.VideoCapture(str(video_path))
    if not video_cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = video_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(args.max_frames, total_frames if total_frames > 0 else args.max_frames)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    runtime_tracker = RuntimeTracker(
        model=model,
        sequence_hw=(height, width),
        assignment_protocol="object-max",
        miss_tolerance=30,
        det_thresh=0.5,
        newborn_thresh=0.5,
        id_thresh=0.2,
        dtype=dtype,
    )

    per_frame_counts = []
    with torch.inference_mode():
        for _ in tqdm(range(frames_to_process), desc="MOTIP smoke", unit="frame"):
            ret, frame = video_cap.read()
            if not ret:
                break

            frame_tensor = simple_transform(frame, args.max_shorter, args.max_longer, dtype)
            frame_tensor = nested_tensor_from_tensor_list([frame_tensor])
            runtime_tracker.update(frame_tensor)
            track_results = runtime_tracker.get_track_results()
            boxes = track_results.get("bbox", [])
            ids = track_results.get("id", [])
            per_frame_counts.append(int(len(ids)))

            for bbox, obj_id in zip(boxes, ids):
                x, y, w, h = map(int, bbox.detach().cpu().tolist())
                obj_id_int = int(obj_id)
                color = get_color(obj_id_int, rgb=False, use_int=True)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID: {obj_id_int}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            writer.write(frame)

    video_cap.release()
    writer.release()

    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "video": str(video_path),
        "output": str(output_path),
        "frames_requested": args.max_frames,
        "frames_processed": len(per_frame_counts),
        "tracks_per_frame": per_frame_counts,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }
    summary_path = Path(args.summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
