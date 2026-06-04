from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def extract_detr_pretrain(source: str | Path, output: str | Path) -> dict[str, object]:
    source = Path(source)
    output = Path(output)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_state = checkpoint["model"]
    detr_state = {
        key.removeprefix("detr."): value
        for key, value in source_state.items()
        if key.startswith("detr.")
    }
    if not detr_state:
        raise ValueError(f"No detr.* keys found in checkpoint: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": detr_state}, output)
    summary = {
        "source": str(source),
        "output": str(output),
        "num_source_keys": len(source_state),
        "num_detr_keys": len(detr_state),
        "has_class_embed": any("class_embed" in key for key in detr_state),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DETR-only pretrain from a MOTIP checkpoint.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(extract_detr_pretrain(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
