#!/usr/bin/env python3
"""Transplant MOTIP tracking weights into a target MOTIP checkpoint.

The main use case is warm-starting tomato MOTIP fine-tuning from an official
tracking-trained MOTIP checkpoint. The DETR detector can stay tomato-specific
via ``--target-base-checkpoint`` while tracking modules are copied from the
official checkpoint. ID vocabulary tensors are allowed to differ in size and can
be copied by overlap instead of silently skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from configs.util import load_super_config
from models.motip import build as build_motip
from utils.misc import yaml_to_dict


DEFAULT_SOURCE_URL = "https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_bft.pth"
WORD_TO_EMBED_KEY = "id_decoder.word_to_embed.weight"
EMBED_TO_WORD_PREFIX = "id_decoder.embed_to_word_layers."


@dataclass
class TransferRecord:
    key: str
    action: str
    source_shape: list[int] | None = None
    target_shape: list[int] | None = None
    copied_primary: int | None = None
    copied_unknown: bool | None = None
    reason: str | None = None


@dataclass
class TransferReport:
    name: str
    source: str
    include_prefixes: list[str]
    exclude_prefixes: list[str]
    vocab_policy: str
    copy_unknown: bool
    copied_exact: list[TransferRecord] = field(default_factory=list)
    copied_partial: list[TransferRecord] = field(default_factory=list)
    skipped: list[TransferRecord] = field(default_factory=list)
    missing_in_target: list[TransferRecord] = field(default_factory=list)
    excluded: list[TransferRecord] = field(default_factory=list)
    outside_include: int = 0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "copied_exact": len(self.copied_exact),
            "copied_partial": len(self.copied_partial),
            "skipped": len(self.skipped),
            "missing_in_target": len(self.missing_in_target),
            "excluded": len(self.excluded),
            "outside_include": self.outside_include,
        }


def _shape(tensor: torch.Tensor) -> list[int]:
    return list(tensor.shape)


def is_vocab_tensor(key: str) -> bool:
    return key == WORD_TO_EMBED_KEY or (
        key.startswith(EMBED_TO_WORD_PREFIX) and key.endswith(".weight")
    )


def _matches_prefix(key: str, prefixes: Iterable[str]) -> bool:
    prefixes = list(prefixes)
    return not prefixes or any(key.startswith(prefix) for prefix in prefixes)


def _partial_copy_vocab_tensor(
    key: str,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    copy_unknown: bool,
) -> tuple[torch.Tensor, TransferRecord] | tuple[None, TransferRecord]:
    """Copy the overlapping ID vocabulary area for known MOTIP vocab tensors."""
    if source.ndim != 2 or target.ndim != 2:
        return None, TransferRecord(
            key=key,
            action="skip",
            source_shape=_shape(source),
            target_shape=_shape(target),
            reason="vocab tensor is not 2D",
        )

    output = target.clone()
    copied_unknown = False
    if key == WORD_TO_EMBED_KEY:
        if source.shape[0] != target.shape[0]:
            return None, TransferRecord(
                key=key,
                action="skip",
                source_shape=_shape(source),
                target_shape=_shape(target),
                reason="embedding dim mismatch",
            )
        known = min(source.shape[1] - 1, target.shape[1] - 1)
        if known > 0:
            output[:, :known] = source[:, :known]
        if copy_unknown:
            output[:, -1] = source[:, -1]
            copied_unknown = True
    else:
        if source.shape[1] != target.shape[1]:
            return None, TransferRecord(
                key=key,
                action="skip",
                source_shape=_shape(source),
                target_shape=_shape(target),
                reason="embedding dim mismatch",
            )
        known = min(source.shape[0] - 1, target.shape[0] - 1)
        if known > 0:
            output[:known, :] = source[:known, :]
        if copy_unknown:
            output[-1, :] = source[-1, :]
            copied_unknown = True

    return output, TransferRecord(
        key=key,
        action="copy_partial_vocab",
        source_shape=_shape(source),
        target_shape=_shape(target),
        copied_primary=known,
        copied_unknown=copied_unknown,
    )


def apply_weight_transfer(
    *,
    name: str,
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    source_label: str,
    include_prefixes: list[str] | None = None,
    exclude_prefixes: list[str] | None = None,
    vocab_policy: str = "copy-overlap",
    copy_unknown: bool = True,
    dry_run: bool = False,
) -> TransferReport:
    include_prefixes = include_prefixes or []
    exclude_prefixes = exclude_prefixes or []
    report = TransferReport(
        name=name,
        source=source_label,
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
        vocab_policy=vocab_policy,
        copy_unknown=copy_unknown,
    )

    for key, source_tensor in source_state.items():
        if not _matches_prefix(key, include_prefixes):
            report.outside_include += 1
            continue
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            report.excluded.append(
                TransferRecord(key=key, action="exclude", source_shape=_shape(source_tensor))
            )
            continue
        if key not in target_state:
            report.missing_in_target.append(
                TransferRecord(key=key, action="missing", source_shape=_shape(source_tensor))
            )
            continue

        target_tensor = target_state[key]
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            if not dry_run:
                target_state[key] = source_tensor.detach().cpu().clone()
            report.copied_exact.append(
                TransferRecord(
                    key=key,
                    action="copy_exact",
                    source_shape=_shape(source_tensor),
                    target_shape=_shape(target_tensor),
                )
            )
            continue

        if vocab_policy == "copy-overlap" and is_vocab_tensor(key):
            copied, record = _partial_copy_vocab_tensor(
                key,
                source_tensor.detach().cpu(),
                target_tensor.detach().cpu(),
                copy_unknown=copy_unknown,
            )
            if copied is None:
                report.skipped.append(record)
            else:
                if not dry_run:
                    target_state[key] = copied
                report.copied_partial.append(record)
            continue

        report.skipped.append(
            TransferRecord(
                key=key,
                action="skip_shape_mismatch",
                source_shape=_shape(source_tensor),
                target_shape=_shape(target_tensor),
                reason="shape mismatch",
            )
        )

    return report


def checkpoint_model_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Checkpoint '{path}' does not contain a top-level 'model' state.")
    return checkpoint["model"]


def ensure_checkpoint(path: Path, source_url: str | None) -> Path:
    if path.exists():
        return path
    if not source_url:
        raise FileNotFoundError(f"Source checkpoint does not exist: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".download")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"[download] {source_url} -> {path}", file=sys.stderr)
    try:
        with urllib.request.urlopen(source_url) as response, tmp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return path


def build_target_state(config_path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    config = yaml_to_dict(str(config_path))
    config = load_super_config(config, config["SUPER_CONFIG_PATH"])
    model, _criterion = build_motip(config=config)
    return config, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _records_to_json(records: list[TransferRecord]) -> list[dict]:
    return [asdict(record) for record in records]


def _report_to_json(report: TransferReport) -> dict:
    data = asdict(report)
    data["summary"] = report.summary()
    for key in ("copied_exact", "copied_partial", "skipped", "missing_in_target", "excluded"):
        data[key] = _records_to_json(getattr(report, key))
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a target MOTIP checkpoint with selected tracking weights transplanted."
    )
    parser.add_argument("--source", type=Path, required=True, help="Official/source MOTIP checkpoint.")
    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help="Download URL used when --source is missing. Set empty string to disable.",
    )
    parser.add_argument("--target-config", type=Path, required=True, help="Target MOTIP config.")
    parser.add_argument(
        "--target-base-checkpoint",
        type=Path,
        help="Optional tomato/base checkpoint used to initialize compatible target weights.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output full target checkpoint.")
    parser.add_argument("--report-json", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=None,
        help="Prefix copied from source. Repeatable. Defaults to trajectory_modeling. and id_decoder.",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Prefix excluded from source transfer. Repeatable.",
    )
    parser.add_argument(
        "--vocab-policy",
        choices=("copy-overlap", "skip"),
        default="copy-overlap",
        help="How to handle ID vocabulary tensors with different shapes.",
    )
    parser.add_argument("--no-copy-unknown", action="store_true", help="Do not copy the source unknown ID row/column.")
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing the output checkpoint.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    include_prefixes = args.include_prefix or ["trajectory_modeling.", "id_decoder."]
    source_url = args.source_url or None
    copy_unknown = not args.no_copy_unknown

    source_path = ensure_checkpoint(args.source, source_url)
    config, target_state = build_target_state(args.target_config)
    reports: list[TransferReport] = []

    if args.target_base_checkpoint is not None:
        base_state = checkpoint_model_state(args.target_base_checkpoint)
        reports.append(
            apply_weight_transfer(
                name="target_base",
                source_state=base_state,
                target_state=target_state,
                source_label=str(args.target_base_checkpoint),
                include_prefixes=[],
                exclude_prefixes=[],
                vocab_policy=args.vocab_policy,
                copy_unknown=copy_unknown,
                dry_run=args.dry_run,
            )
        )

    source_state = checkpoint_model_state(source_path)
    reports.append(
        apply_weight_transfer(
            name="official_tracking",
            source_state=source_state,
            target_state=target_state,
            source_label=str(source_path),
            include_prefixes=include_prefixes,
            exclude_prefixes=args.exclude_prefix,
            vocab_policy=args.vocab_policy,
            copy_unknown=copy_unknown,
            dry_run=args.dry_run,
        )
    )

    report_json = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z%z"),
        "source": str(source_path),
        "source_url": source_url,
        "target_config": str(args.target_config),
        "target_base_checkpoint": str(args.target_base_checkpoint) if args.target_base_checkpoint else None,
        "output": str(args.output),
        "dry_run": args.dry_run,
        "target_config_resolved": {
            key: config.get(key)
            for key in (
                "SAMPLE_LENGTHS",
                "SAMPLE_INTERVALS",
                "NUM_ID_VOCABULARY",
                "NUM_TRAINING_IDS",
                "PSEUDOMOT_SUB_DIR",
                "OUTPUTS_DIR",
                "EXP_NAME",
            )
        },
        "reports": [_report_to_json(report) for report in reports],
        "summaries": [report.summary() for report in reports],
    }

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report_json, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"summaries": report_json["summaries"], "dry_run": args.dry_run}, indent=2))

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": target_state,
                "optimizer": None,
                "scheduler": None,
                "states": {"start_epoch": 0, "global_step": 0},
                "meta": {
                    "source": str(source_path),
                    "source_url": source_url,
                    "target_config": str(args.target_config),
                    "target_base_checkpoint": str(args.target_base_checkpoint)
                    if args.target_base_checkpoint
                    else None,
                    "transfer_summaries": report_json["summaries"],
                },
            },
            args.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
