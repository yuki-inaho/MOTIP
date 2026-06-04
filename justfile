set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
venv := "/home/kasm-user/Desktop/MOTIP/.venv"
tracklet_coco := "/home/kasm-user/Desktop/tomato_tracking_deim_mot/outputs/nyx660_jun04/coco_good.json"
tracklet_images := "/home/kasm-user/Desktop/NYX660_2025_12_01_17_33_27_0135/Color"
tracklet_dataset := "datasets/TomatoTrackletMOT"
tracklet_config := "configs/train_tracklet_pseudomot_smoke.yaml"
tracklet_full_config := "configs/train_tracklet_pseudomot_full.yaml"
motip_dancetrack_ckpt := "outputs/r50_deformable_detr_motip_dancetrack/r50_deformable_detr_motip_dancetrack.pth"
tracklet_detr_pretrain := "pretrains/r50_deformable_detr_coco_dancetrack.pth"

sync:
    UV_PROJECT_ENVIRONMENT="{{venv}}" uv sync

# Reproducible install from the pinned lock (no implicit re-resolution).
# NOTE: this removes the locally-built MultiScaleDeformableAttention op, so run
# `just build-ops` afterwards (see `just repro-doc`).
sync-frozen:
    UV_PROJECT_ENVIRONMENT="{{venv}}" uv sync --frozen

# Print the reproducible-environment regimen (uv --frozen / --no-sync / seed).
repro-doc:
    @echo "Reproducible environment regimen (see README.md > Reproducible local environment):"
    @echo "  1) export UV_PROJECT_ENVIRONMENT={{venv}}"
    @echo "  2) just sync-frozen     # uv sync --frozen (exact locked deps; fails on lock mismatch)"
    @echo "  3) just build-ops       # rebuild MultiScaleDeformableAttention op (removed by --frozen)"
    @echo "  4) uv run --no-sync ... # run without re-resolving the lock"
    @echo "Seeding is rank-aware: SEED is offset by distributed_rank() in utils/misc.py:set_seed."
    @echo "Generic overrides: -u KEY=VALUE (YAML-typed); unknown keys fail loudly (no silent fallback)."

env-info:
    UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python -c 'import sys, torch; print("python", sys.version); print("torch", torch.__version__); print("cuda_available", torch.cuda.is_available()); print("cuda_runtime", torch.version.cuda if torch.cuda.is_available() else "n/a"); print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"); print("capability", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "n/a")'

build-ops:
    cd models/ops && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python setup.py build install

test-ops:
    cd models/ops && UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python test.py

test-ops-smoke:
    cd models/ops && UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python -c 'from test import check_forward_equal_with_pytorch_double, check_forward_equal_with_pytorch_float, check_gradient_numerical; check_forward_equal_with_pytorch_double(); check_forward_equal_with_pytorch_float(); check_gradient_numerical(30, True, True, True)'

import-smoke:
    UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python -c 'import torch; from configs.util import load_super_config; from models.motip import build as build_model; from models.runtime_tracker import RuntimeTracker; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("./configs/r50_deformable_detr_motip_dancetrack.yaml"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); print("torch", torch.__version__, "cuda", torch.cuda.is_available()); print("config_keys", len(cfg)); print("build_model", build_model.__name__); print("runtime_tracker", RuntimeTracker.__name__)'

demo-requirements-check:
    UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python -c 'from pathlib import Path; import sys; required = {"config": Path("configs/r50_deformable_detr_motip_dancetrack.yaml"), "checkpoint": Path("outputs/r50_deformable_detr_motip_dancetrack/r50_deformable_detr_motip_dancetrack.pth"), "video": Path("outputs/video_process_demo/hpop_dancers.mp4")}; [print(f"{name}: {path} exists={path.exists()} size={path.stat().st_size if path.exists() else 0}") for name, path in required.items()]; missing = [name for name, path in required.items() if not path.exists()]; sys.exit(f"missing demo assets: {chr(44).join(missing)}" if missing else 0)'

demo-smoke MAX_FRAMES="3":
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python demo/video_process_smoke.py --max-frames "{{MAX_FRAMES}}"

build-tracklet-pseudomot:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python tools/convert_coco_tracklets_to_pseudomot.py --coco "{{tracklet_coco}}" --image-dir "{{tracklet_images}}" --output-root "{{tracklet_dataset}}" --sequence-name nyx660_jun04 --split train

prepare-tracklet-pretrain:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python tools/extract_detr_pretrain.py --source "{{motip_dancetrack_ckpt}}" --output "{{tracklet_detr_pretrain}}"

config-tracklet-smoke:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{tracklet_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["DATA_ROOT", "DATASETS", "DATASET_SPLITS", "PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "EPOCHS", "MAX_TRAIN_STEPS", "DETR_PRETRAIN", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

loader-tracklet-smoke:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python -c 'from data.joint_dataset import JointDataset; ds = JointDataset(data_root="./datasets", datasets=["PseudoMOT"], splits=["train"], pseudomot_sub_dir="TomatoTrackletMOT"); ds.set_sample_details(sample_length=2, sample_interval=1); print(ds.statistics()); print("samples", len(ds))'

train-tracklet-smoke:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python train.py --config-path "{{tracklet_config}}"

# Show the resolved key values for the FULL training config (no MAX_TRAIN_STEPS).
config-tracklet-full:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{tracklet_full_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["DATASETS", "DATASET_SPLITS", "PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "SAMPLE_INTERVALS", "NUM_ID_VOCABULARY", "NUM_TRAINING_IDS", "EPOCHS", "MAX_TRAIN_STEPS", "AMP_DTYPE", "EMA_ENABLED", "TENSORBOARD", "EARLY_STOP", "USE_DECODER_CHECKPOINT", "INFERENCE_DATASET", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

# Launch the FULL tracklet training (background recommended; started in B6, not here).
train-tracklet-full:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{tracklet_full_config}}"

# TensorBoard for the full run (point --logdir at the run's train/tb directory).
tb:
    UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync tensorboard --logdir outputs/tracklet_pseudomot_full/train/tb
