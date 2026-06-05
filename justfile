set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
venv := "/home/kasm-user/Desktop/MOTIP/.venv"
tracklet_coco := "/home/kasm-user/Desktop/tomato_tracking_deim_mot/outputs/nyx660_jun04/coco_good.json"
tracklet_images := "/home/kasm-user/Desktop/NYX660_2025_12_01_17_33_27_0135/Color"
tracklet_dataset := "datasets/TomatoTrackletMOT"
tracklet_5fps_dataset := "datasets/TomatoTrackletMOT_5fps"
tracklet_retrack_optuna_dataset := "datasets/TomatoTrackletMOT_retrack_optuna"
tracklet_config := "configs/train_tracklet_pseudomot_smoke.yaml"
tracklet_full_config := "configs/train_tracklet_pseudomot_full.yaml"
tracklet_5fps_config := "configs/finetune_tracklet_pseudomot_5fps_id16.yaml"
tracklet_retrack_optuna_config := "configs/finetune_tracklet_pseudomot_retrack_optuna.yaml"
tracklet_bft_official_config := "configs/finetune_tracklet_pseudomot_retrack_optuna_bft_official.yaml"
tracklet_bft_official_muon_config := "configs/finetune_tracklet_pseudomot_retrack_optuna_bft_official_muon.yaml"
tracklet_bft_official_schedulefree_config := "configs/finetune_tracklet_pseudomot_retrack_optuna_bft_official_schedulefree.yaml"
applemots_raw := "/home/kasm-user/Desktop/APPLE_MOTS"
applemots_dataset := "datasets/AppleMOTSPseudoMOT"
applemots_coco_dataset := "datasets/AppleMOTSCOCO"
applemots_coco_deim_dataset := "datasets/AppleMOTSCOCO_DEIM"
applemots_smoke_config := "configs/train_applemots_pseudomot_smoke.yaml"
applemots_bft_schedulefree_config := "configs/train_applemots_pseudomot_bft_official_schedulefree.yaml"
applemots_bft_sl8_pretrain_config := "configs/train_applemots_pseudomot_bft_schedulefree_sl8_pretrain.yaml"
applemots_bft_sl8_pretrain_ckpt := "outputs/applemots_pseudomot_bft_schedulefree_sl8_pretrain/checkpoint_3.pth"
applemots_overfit_retrack_dataset := "datasets/AppleMOTSPseudoMOT_overfit_train0000_retrack"
applemots_overfit_retrack_config := "configs/finetune_applemots_overfit_train0000_retrack.yaml"
applemots_overfit_retrack_ckpt := "outputs/applemots_overfit_train0000_retrack_sl8_schedulefree/checkpoint_11.pth"
applemots_overfit_retrack_idonly_config := "configs/finetune_applemots_overfit_train0000_retrack_idonly.yaml"
applemots_overfit_retrack_idonly_ckpt := "outputs/applemots_overfit_train0000_retrack_idonly_sl8_schedulefree/checkpoint_19.pth"
applemots_to_tomato_tracking_pretrain := "pretrains/motip_applemots_tracking_to_tomato_retrack_optuna_sl20.pth"
applemots_to_tomato_tracking_report := "reports/motip_applemots_tracking_to_tomato_retrack_optuna_sl20_transfer_report.json"
motip_dancetrack_ckpt := "outputs/r50_deformable_detr_motip_dancetrack/r50_deformable_detr_motip_dancetrack.pth"
motip_bft_ckpt := "outputs/r50_deformable_detr_motip_bft/r50_deformable_detr_motip_bft.pth"
motip_bft_url := "https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_bft.pth"
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

build-tracklet-pseudomot-5fps:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/convert_coco_tracklets_to_pseudomot.py \
        --coco "{{tracklet_coco}}" --image-dir "{{tracklet_images}}" \
        --output-root "{{tracklet_5fps_dataset}}" --sequence-name nyx660_jun04_stride6 \
        --split train --frame-rate 5 --frame-stride 6

build-tracklet-pseudomot-retrack-optuna:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/convert_track_json_to_pseudomot.py \
        --track-json "outputs/tracklet_pseudomot_5fps_id16/retrack_optuna/tracks.json" \
        --image-dir "{{tracklet_seq_images}}" \
        --output-root "{{tracklet_retrack_optuna_dataset}}" --sequence-name nyx660_jun04_retrack_optuna \
        --split train --frame-rate 30

build-applemots-pseudomot:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/convert_apple_mots_to_pseudomot.py \
        --applemots-root "{{applemots_raw}}" --output-root "{{applemots_dataset}}" \
        --splits train testing --frame-rate 30

build-applemots-coco:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/convert_apple_mots_to_coco.py \
        --applemots-root "{{applemots_raw}}" --output-root "{{applemots_coco_dataset}}" \
        --splits train testing

build-applemots-coco-deim:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/convert_apple_mots_to_coco.py \
        --applemots-root "{{applemots_raw}}" --output-root "{{applemots_coco_deim_dataset}}" \
        --splits train testing --category-id 0

# Build the tiny AppleMOTS train/0000 overfit dataset from the Optuna center-distance retrack teacher.
build-applemots-overfit-retrack-train0000:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/convert_track_json_to_pseudomot.py \
        --track-json "outputs/applemots_pseudomot_bft_schedulefree_sl8_pretrain/retrack_overfit_train0000_det020_onlydetr_center/tracks.json" \
        --image-dir "datasets/AppleMOTSPseudoMOT/train/0000/img1" \
        --output-root "{{applemots_overfit_retrack_dataset}}" \
        --sequence-name "applemots_train0000_retrack_center" \
        --split train --frame-rate 30

summary-applemots-coco:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'import json; from pathlib import Path; s=json.loads(Path("{{applemots_coco_dataset}}/conversion_summary.json").read_text()); print(json.dumps(s["by_split"], indent=2)); print("train_ann", s["split_summaries"][0]["ann_file"]); print("testing_ann", s["split_summaries"][1]["ann_file"])'

summary-applemots-coco-deim:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'import json; from pathlib import Path; s=json.loads(Path("{{applemots_coco_deim_dataset}}/conversion_summary.json").read_text()); print(json.dumps(s["by_split"], indent=2)); print("category_id", s["category_id"]); print("train_ann", s["split_summaries"][0]["ann_file"]); print("testing_ann", s["split_summaries"][1]["ann_file"])'

prepare-tracklet-pretrain:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python tools/extract_detr_pretrain.py --source "{{motip_dancetrack_ckpt}}" --output "{{tracklet_detr_pretrain}}"

config-tracklet-smoke:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{tracklet_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["DATA_ROOT", "DATASETS", "DATASET_SPLITS", "PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "EPOCHS", "MAX_TRAIN_STEPS", "DETR_PRETRAIN", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

loader-tracklet-smoke:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python -c 'from data.joint_dataset import JointDataset; ds = JointDataset(data_root="./datasets", datasets=["PseudoMOT"], splits=["train"], pseudomot_sub_dir="TomatoTrackletMOT"); ds.set_sample_details(sample_length=2, sample_interval=1); print(ds.statistics()); print("samples", len(ds))'

loader-tracklet-5fps:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from data.joint_dataset import JointDataset; ds = JointDataset(data_root="./datasets", datasets=["PseudoMOT"], splits=["train"], pseudomot_sub_dir="TomatoTrackletMOT_5fps"); ds.set_sample_details(sample_length=16, sample_interval=1); print(ds.statistics()); print("samples", len(ds))'

loader-tracklet-retrack-optuna:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from data.joint_dataset import JointDataset; ds = JointDataset(data_root="./datasets", datasets=["PseudoMOT"], splits=["train"], pseudomot_sub_dir="TomatoTrackletMOT_retrack_optuna"); ds.set_sample_details(sample_length=8, sample_interval=1); print(ds.statistics()); print("samples", len(ds))'

loader-applemots:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from data.joint_dataset import JointDataset; ds = JointDataset(data_root="./datasets", datasets=["PseudoMOT"], splits=["train"], pseudomot_sub_dir="AppleMOTSPseudoMOT"); ds.set_sample_details(sample_length=2, sample_interval=1); print(ds.statistics()); print("samples", len(ds))'

loader-applemots-overfit-retrack:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from data.joint_dataset import JointDataset; ds = JointDataset(data_root="./datasets", datasets=["PseudoMOT"], splits=["train"], pseudomot_sub_dir="AppleMOTSPseudoMOT_overfit_train0000_retrack"); ds.set_sample_details(sample_length=8, sample_interval=1); print(ds.statistics()); print("samples", len(ds))'

train-tracklet-smoke:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run python train.py --config-path "{{tracklet_config}}"

# Show the resolved key values for the FULL training config (no MAX_TRAIN_STEPS).
config-tracklet-full:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{tracklet_full_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["DATASETS", "DATASET_SPLITS", "PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "SAMPLE_INTERVALS", "NUM_ID_VOCABULARY", "NUM_TRAINING_IDS", "EPOCHS", "MAX_TRAIN_STEPS", "AMP_DTYPE", "EMA_ENABLED", "TENSORBOARD", "EARLY_STOP", "USE_DECODER_CHECKPOINT", "INFERENCE_DATASET", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

# Show the resolved key values for the 5FPS ID fine-tune config.
config-tracklet-5fps:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{tracklet_5fps_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["DATASETS", "DATASET_SPLITS", "PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "SAMPLE_INTERVALS", "NUM_ID_VOCABULARY", "NUM_TRAINING_IDS", "EPOCHS", "MAX_TRAIN_STEPS", "AMP_DTYPE", "EMA_ENABLED", "TENSORBOARD", "EARLY_STOP", "AUG_NUM_GROUPS", "ID_LOSS_WEIGHT", "DETR_PRETRAIN", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

# Show resolved key values for the Optuna-retracked pseudo-label fine-tune config.
config-tracklet-retrack-optuna:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{tracklet_retrack_optuna_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "SAMPLE_INTERVALS", "NUM_ID_VOCABULARY", "NUM_TRAINING_IDS", "RESUME_MODEL", "RESUME_OPTIMIZER", "RESUME_SCHEDULER", "EPOCHS", "SCHEDULER_MILESTONES", "MAX_TRAIN_STEPS", "AMP_DTYPE", "EMA_ENABLED", "AUG_NUM_GROUPS", "ID_LOSS_WEIGHT", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

# Show resolved key values for the BFT-official tracking-transfer fine-tune config.
config-tracklet-bft-official:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{tracklet_bft_official_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "SAMPLE_INTERVALS", "REL_PE_LENGTH", "MISS_TOLERANCE", "AUG_RESIZE_SCALES", "AUG_MAX_SIZE", "AUG_RANDOM_CROP_PROB", "AUG_NUM_GROUPS", "NUM_ID_VOCABULARY", "NUM_TRAINING_IDS", "OPTIMIZER_TYPE", "SCHEDULER_TYPE", "LR_WARMUP_EPOCHS", "RESUME_MODEL", "RESUME_OPTIMIZER", "RESUME_SCHEDULER", "EPOCHS", "SCHEDULER_MILESTONES", "MAX_TRAIN_STEPS", "AMP_DTYPE", "EMA_ENABLED", "EARLY_STOP", "ID_LOSS_WEIGHT", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

# Show resolved key values for BFT-official + Muon.
config-tracklet-bft-official-muon:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{tracklet_bft_official_muon_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["OPTIMIZER_TYPE", "SCHEDULER_TYPE", "LR_WARMUP_EPOCHS", "MUON_LR", "MUON_ADAM_LR", "PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "SAMPLE_INTERVALS", "NUM_ID_VOCABULARY", "NUM_TRAINING_IDS", "EPOCHS", "EARLY_STOP", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

# Show resolved key values for BFT-official + AdamW ScheduleFree.
config-tracklet-bft-official-schedulefree:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{tracklet_bft_official_schedulefree_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["OPTIMIZER_TYPE", "SCHEDULER_TYPE", "SCHEDULEFREE_LR", "SCHEDULEFREE_WEIGHT_DECAY", "LR_WARMUP_EPOCHS", "PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "SAMPLE_INTERVALS", "NUM_ID_VOCABULARY", "NUM_TRAINING_IDS", "EPOCHS", "EARLY_STOP", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

config-applemots-smoke:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{applemots_smoke_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["DATASETS", "DATASET_SPLITS", "PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "SAMPLE_INTERVALS", "NUM_ID_VOCABULARY", "NUM_TRAINING_IDS", "MAX_TRAIN_STEPS", "DETR_PRETRAIN", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

config-applemots-bft-schedulefree:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{applemots_bft_schedulefree_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["DATASETS", "DATASET_SPLITS", "PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "SAMPLE_INTERVALS", "REL_PE_LENGTH", "MISS_TOLERANCE", "AUG_RESIZE_SCALES", "AUG_MAX_SIZE", "AUG_NUM_GROUPS", "NUM_ID_VOCABULARY", "NUM_TRAINING_IDS", "OPTIMIZER_TYPE", "SCHEDULER_TYPE", "RESUME_MODEL", "EPOCHS", "MAX_TRAIN_STEPS", "AMP_DTYPE", "EMA_ENABLED", "EARLY_STOP", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

config-applemots-bft-sl8-pretrain:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{applemots_bft_sl8_pretrain_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["DATASETS", "DATASET_SPLITS", "PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "SAMPLE_INTERVALS", "REL_PE_LENGTH", "MISS_TOLERANCE", "AUG_RESIZE_SCALES", "AUG_MAX_SIZE", "AUG_NUM_GROUPS", "NUM_ID_VOCABULARY", "NUM_TRAINING_IDS", "OPTIMIZER_TYPE", "SCHEDULER_TYPE", "RESUME_MODEL", "EPOCHS", "MAX_TRAIN_STEPS", "AMP_DTYPE", "EMA_ENABLED", "EARLY_STOP", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

config-applemots-overfit-retrack:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{applemots_overfit_retrack_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "SAMPLE_INTERVALS", "NUM_ID_VOCABULARY", "NUM_TRAINING_IDS", "RESUME_MODEL", "RESUME_OPTIMIZER", "RESUME_SCHEDULER", "EPOCHS", "MAX_TRAIN_STEPS", "OPTIMIZER_TYPE", "SCHEDULER_TYPE", "AMP_DTYPE", "EMA_ENABLED", "EARLY_STOP", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

config-applemots-overfit-retrack-idonly:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python -c 'from configs.util import load_super_config; from utils.misc import yaml_to_dict; cfg = yaml_to_dict("{{applemots_overfit_retrack_idonly_config}}"); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"]); keys = ["PSEUDOMOT_SUB_DIR", "SAMPLE_LENGTHS", "SAMPLE_INTERVALS", "DETR_NUM_TRAIN_FRAMES", "ID_LOSS_WEIGHT", "NUM_ID_VOCABULARY", "NUM_TRAINING_IDS", "RESUME_MODEL", "RESUME_OPTIMIZER", "RESUME_SCHEDULER", "EPOCHS", "MAX_TRAIN_STEPS", "OPTIMIZER_TYPE", "SCHEDULER_TYPE", "AMP_DTYPE", "EMA_ENABLED", "EARLY_STOP", "OUTPUTS_DIR", "EXP_NAME"]; [print(f"{key}={cfg.get(key)}") for key in keys]'

# Launch the FULL tracklet training (background recommended; started in B6, not here).
train-tracklet-full:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{tracklet_full_config}}"

# Launch the 5FPS ID fine-tune. Use PYTORCH_CUDA_ALLOC_CONF to reduce allocator fragmentation.
train-tracklet-5fps:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{tracklet_5fps_config}}"

# Fine-tune from checkpoint_39 on Optuna-retracked pseudo labels.
train-tracklet-retrack-optuna:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{tracklet_retrack_optuna_config}}"

# Fine-tune with BFT-official tracking weights and BFT-style temporal sampling.
train-tracklet-bft-official:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{tracklet_bft_official_config}}"

# Fine-tune with BFT-official tracking weights and Muon optimizer.
train-tracklet-bft-official-muon:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{tracklet_bft_official_muon_config}}"

# Fine-tune with BFT-official tracking weights and AdamW ScheduleFree optimizer.
train-tracklet-bft-official-schedulefree:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{tracklet_bft_official_schedulefree_config}}"

train-applemots-smoke:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{applemots_smoke_config}}"

transplant-applemots-bft-schedulefree:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/transplant_motip_tracking_weights.py \
        --source "{{motip_bft_ckpt}}" --source-url "{{motip_bft_url}}" \
        --target-config "{{applemots_bft_schedulefree_config}}" \
        --target-base-checkpoint "{{motip_bft_ckpt}}" \
        --output "pretrains/motip_bft_tracking_to_applemots_sl20.pth" \
        --report-json "reports/motip_bft_tracking_to_applemots_sl20_transfer_report.json"

train-applemots-bft-schedulefree:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{applemots_bft_schedulefree_config}}"

train-applemots-bft-schedulefree-smoke STEPS="60":
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{applemots_bft_schedulefree_config}}" -u MAX_TRAIN_STEPS="{{STEPS}}" EPOCHS=1 OUTPUTS_DIR="./outputs/applemots_pseudomot_bft_official_schedulefree_smoke" EXP_NAME="applemots_pseudomot_bft_official_schedulefree_smoke"

train-applemots-bft-sl8-pretrain:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{applemots_bft_sl8_pretrain_config}}"

train-applemots-overfit-retrack:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{applemots_overfit_retrack_config}}"

train-applemots-overfit-retrack-idonly:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{applemots_overfit_retrack_idonly_config}}"

transplant-applemots-to-tomato:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/transplant_motip_tracking_weights.py \
        --source "{{applemots_bft_sl8_pretrain_ckpt}}" --source-url "" \
        --target-config "{{tracklet_bft_official_schedulefree_config}}" \
        --target-base-checkpoint "outputs/tracklet_pseudomot_retrack_optuna_ft/checkpoint_40.pth" \
        --output "{{applemots_to_tomato_tracking_pretrain}}" \
        --report-json "{{applemots_to_tomato_tracking_report}}" \
        --include-prefix "trajectory_modeling." --include-prefix "id_decoder."

# Smoke fine-tune tomato retrack-optuna pseudo labels from the AppleMOTS tracking pretrain.
# NOTE: runtime_option.py defines -u as nargs="+", so all overrides must be in one -u group.
train-tracklet-applemots-transfer-smoke STEPS="60":
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python train.py --config-path "{{tracklet_bft_official_schedulefree_config}}" \
        -u RESUME_MODEL="{{applemots_to_tomato_tracking_pretrain}}" OUTPUTS_DIR="./outputs/tracklet_pseudomot_retrack_optuna_applemots_bft_schedulefree_smoke" EXP_NAME="tracklet_pseudomot_retrack_optuna_applemots_bft_schedulefree_smoke" MAX_TRAIN_STEPS="{{STEPS}}" EPOCHS=1 SAVE_CHECKPOINT_PER_EPOCH=1

# TensorBoard for the full run (point --logdir at the run's train/tb directory).
tb:
    UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync tensorboard --logdir outputs/tracklet_pseudomot_full/train/tb

# --- Tracklet inference & visualization ---
tracklet_full_ckpt := "outputs/tracklet_pseudomot_full/checkpoint_7.pth"
tracklet_seq_images := "datasets/TomatoTrackletMOT/train/nyx660_jun04/img1"
tracklet_infer_json := "outputs/tracklet_pseudomot_full/infer/tracks.json"
tracklet_5fps_ckpt := "outputs/tracklet_pseudomot_5fps_id16/checkpoint_39.pth"
tracklet_5fps_infer_json := "outputs/tracklet_pseudomot_5fps_id16/infer_30fps/tracks.json"
tracklet_5fps_comparison_json := "outputs/tracklet_pseudomot_5fps_id16/infer_30fps/comparison.json"
tracklet_5fps_retrack_json := "outputs/tracklet_pseudomot_5fps_id16/retrack_bytetrack/tracks.json"
tracklet_5fps_retrack_mot := "outputs/tracklet_pseudomot_5fps_id16/retrack_bytetrack/tracks_mot.txt"
tracklet_5fps_retrack_summary := "outputs/tracklet_pseudomot_5fps_id16/retrack_bytetrack/summary.json"
tracklet_5fps_optuna_retrack_json := "outputs/tracklet_pseudomot_5fps_id16/retrack_optuna/tracks.json"
tracklet_5fps_optuna_retrack_mot := "outputs/tracklet_pseudomot_5fps_id16/retrack_optuna/tracks_mot.txt"
tracklet_5fps_optuna_retrack_summary := "outputs/tracklet_pseudomot_5fps_id16/retrack_optuna/summary.json"
tracklet_5fps_optuna_study := "outputs/tracklet_pseudomot_5fps_id16/retrack_optuna/optuna_study.json"
tracklet_retrack_optuna_ft_ckpt := "outputs/tracklet_pseudomot_retrack_optuna_ft/checkpoint_40.pth"
tracklet_retrack_optuna_ft_infer_json := "outputs/tracklet_pseudomot_retrack_optuna_ft/infer_30fps/tracks.json"
tracklet_retrack_optuna_ft_comparison_json := "outputs/tracklet_pseudomot_retrack_optuna_ft/infer_30fps/comparison_vs_optuna_retrack.json"
tracklet_bft_tracking_pretrain := "pretrains/motip_bft_tracking_to_tomato_retrack_optuna_sl20.pth"
tracklet_bft_tracking_report := "pretrains/motip_bft_tracking_to_tomato_retrack_optuna_sl20_report.json"
tracklet_bft_official_ckpt := "outputs/tracklet_pseudomot_retrack_optuna_bft_official_ft/checkpoint_0.pth"
tracklet_bft_official_infer_json := "outputs/tracklet_pseudomot_retrack_optuna_bft_official_ft/infer_30fps/tracks.json"
tracklet_bft_official_comparison_json := "outputs/tracklet_pseudomot_retrack_optuna_bft_official_ft/infer_30fps/comparison_vs_optuna_retrack.json"
tracklet_applemots_transfer_ckpt := "outputs/tracklet_pseudomot_retrack_optuna_applemots_bft_schedulefree_smoke/checkpoint_0.pth"
tracklet_applemots_transfer_infer_json := "outputs/tracklet_pseudomot_retrack_optuna_applemots_bft_schedulefree_smoke/infer_30fps/tracks.json"
tracklet_applemots_transfer_comparison_json := "outputs/tracklet_pseudomot_retrack_optuna_applemots_bft_schedulefree_smoke/infer_30fps/comparison_vs_optuna_retrack.json"

# Create a full tomato-target checkpoint with official BFT MOTIP tracking modules
# transplanted. BFT source is downloaded if missing; tomato detector-compatible
# weights come from the retrack-optuna checkpoint_40 target base.
transplant-motip-bft-tracking:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/transplant_motip_tracking_weights.py \
        --source "{{motip_bft_ckpt}}" --source-url "{{motip_bft_url}}" \
        --target-config "{{tracklet_bft_official_config}}" \
        --target-base-checkpoint "{{tracklet_retrack_optuna_ft_ckpt}}" \
        --output "{{tracklet_bft_tracking_pretrain}}" \
        --report-json "{{tracklet_bft_tracking_report}}" \
        --include-prefix "trajectory_modeling." --include-prefix "id_decoder."

# Run MOTIP tracking inference on the tomato sequence -> JSON (+ MOTChallenge txt). MAX=0 = all frames.
infer-tracklet-full MAX="0" DTYPE="fp32":
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/infer_tracklet.py \
        --config "{{tracklet_full_config}}" --checkpoint "{{tracklet_full_ckpt}}" --use-ema \
        --image-dir "{{tracklet_seq_images}}" --output-json "{{tracklet_infer_json}}" \
        --output-mot "outputs/tracklet_pseudomot_full/infer/tracks_mot.txt" \
        --max-frames "{{MAX}}" --dtype "{{DTYPE}}"

# Run 5FPS fine-tuned checkpoint over the original 30FPS image sequence -> JSON + MOT txt.
infer-tracklet-5fps MAX="0" DTYPE="fp32" CKPT=tracklet_5fps_ckpt:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/infer_tracklet.py \
        --config "{{tracklet_5fps_config}}" --checkpoint "{{CKPT}}" --use-ema \
        --image-dir "{{tracklet_seq_images}}" --output-json "{{tracklet_5fps_infer_json}}" \
        --output-mot "outputs/tracklet_pseudomot_5fps_id16/infer_30fps/tracks_mot.txt" \
        --max-frames "{{MAX}}" --dtype "{{DTYPE}}"

# Run retrack-Optuna pseudo-label fine-tuned checkpoint over the original 30FPS image sequence.
infer-tracklet-retrack-optuna-ft MAX="0" DTYPE="fp32" CKPT=tracklet_retrack_optuna_ft_ckpt:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/infer_tracklet.py \
        --config "{{tracklet_retrack_optuna_config}}" --checkpoint "{{CKPT}}" --use-ema \
        --image-dir "{{tracklet_seq_images}}" --output-json "{{tracklet_retrack_optuna_ft_infer_json}}" \
        --output-mot "outputs/tracklet_pseudomot_retrack_optuna_ft/infer_30fps/tracks_mot.txt" \
        --max-frames "{{MAX}}" --dtype "{{DTYPE}}"

# Run BFT-official tracking-transfer checkpoint over the original 30FPS image sequence.
infer-tracklet-bft-official MAX="0" DTYPE="fp32" CKPT=tracklet_bft_official_ckpt:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/infer_tracklet.py \
        --config "{{tracklet_bft_official_config}}" --checkpoint "{{CKPT}}" --use-ema \
        --image-dir "{{tracklet_seq_images}}" --output-json "{{tracklet_bft_official_infer_json}}" \
        --output-mot "outputs/tracklet_pseudomot_retrack_optuna_bft_official_ft/infer_30fps/tracks_mot.txt" \
        --max-frames "{{MAX}}" --dtype "{{DTYPE}}"

# Run the AppleMOTS train/0000 ID-only overfit checkpoint over its training frames.
infer-applemots-overfit-retrack-idonly-train0000 MAX="0" DTYPE="fp32" CKPT=applemots_overfit_retrack_idonly_ckpt:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/infer_tracklet.py \
        --config "{{applemots_overfit_retrack_idonly_config}}" --checkpoint "{{CKPT}}" --use-ema \
        --image-dir "datasets/AppleMOTSPseudoMOT/train/0000/img1" \
        --output-json "outputs/applemots_overfit_train0000_retrack_idonly_sl8_schedulefree/infer_train0000/tracks.json" \
        --output-mot "outputs/applemots_overfit_train0000_retrack_idonly_sl8_schedulefree/infer_train0000/tracks_mot.txt" \
        --max-shorter 384 --max-longer 1024 --max-frames "{{MAX}}" --dtype "{{DTYPE}}"

# Render tracking JSON -> annotated frames + mp4. MAX=0 = all, FPS default 3.
visualize-tracklet-full MAX="0" FPS="3":
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/visualize_tracks.py \
        --tracks-json "{{tracklet_infer_json}}" --image-dir "{{tracklet_seq_images}}" \
        --output-dir "outputs/tracklet_pseudomot_full/infer/viz" \
        --output-video "outputs/tracklet_pseudomot_full/infer/tracks.mp4" --fps "{{FPS}}" --show-score \
        --max-frames "{{MAX}}"

# Render tracking JSON -> mp4 ONLY (no per-frame dump, saves disk). FPS default 3, MAX=0 = all.
video-tracklet-full FPS="3" MAX="0":
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/visualize_tracks.py \
        --tracks-json "{{tracklet_infer_json}}" --image-dir "{{tracklet_seq_images}}" \
        --output-dir "" \
        --output-video "outputs/tracklet_pseudomot_full/infer/tracks.mp4" --fps "{{FPS}}" --show-score \
        --max-frames "{{MAX}}"

# Render 5FPS fine-tuned tracking JSON -> mp4 ONLY. FPS default 3, MAX=0 = all.
video-tracklet-5fps FPS="3" MAX="0":
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/visualize_tracks.py \
        --tracks-json "{{tracklet_5fps_infer_json}}" --image-dir "{{tracklet_seq_images}}" \
        --output-dir "" \
        --output-video "outputs/tracklet_pseudomot_5fps_id16/infer_30fps/tracks.mp4" --fps "{{FPS}}" --show-score \
        --max-frames "{{MAX}}"

# Render retrack-Optuna fine-tuned tracking JSON -> mp4 only.
video-tracklet-retrack-optuna-ft FPS="3" MAX="0":
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/visualize_tracks.py \
        --tracks-json "{{tracklet_retrack_optuna_ft_infer_json}}" --image-dir "{{tracklet_seq_images}}" \
        --output-dir "" \
        --output-video "outputs/tracklet_pseudomot_retrack_optuna_ft/infer_30fps/tracks.mp4" \
        --fps "{{FPS}}" --show-score --max-frames "{{MAX}}"

# Render BFT-official tracking-transfer inference JSON -> mp4 only.
video-tracklet-bft-official FPS="3" MAX="0":
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/visualize_tracks.py \
        --tracks-json "{{tracklet_bft_official_infer_json}}" --image-dir "{{tracklet_seq_images}}" \
        --output-dir "" \
        --output-video "outputs/tracklet_pseudomot_retrack_optuna_bft_official_ft/infer_30fps/tracks.mp4" \
        --fps "{{FPS}}" --show-score --max-frames "{{MAX}}"

# Compare old full-run tracking JSON and new 5FPS fine-tuned tracking JSON.
compare-tracklet-5fps:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/compare_track_json.py \
        --old-json "{{tracklet_infer_json}}" --new-json "{{tracklet_5fps_infer_json}}" \
        --output-json "{{tracklet_5fps_comparison_json}}"

# Compare Optuna re-tracked pseudo labels and MOTIP output after fine-tuning on those labels.
compare-tracklet-retrack-optuna-ft:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/compare_track_json.py \
        --old-json "{{tracklet_5fps_optuna_retrack_json}}" --new-json "{{tracklet_retrack_optuna_ft_infer_json}}" \
        --output-json "{{tracklet_retrack_optuna_ft_comparison_json}}"

# Compare Optuna re-tracked pseudo labels and BFT tracking-transfer MOTIP output.
compare-tracklet-bft-official:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/compare_track_json.py \
        --old-json "{{tracklet_5fps_optuna_retrack_json}}" --new-json "{{tracklet_bft_official_infer_json}}" \
        --output-json "{{tracklet_bft_official_comparison_json}}"

# Run the AppleMOTS-pretrained tomato transfer checkpoint over the original 30FPS image sequence.
infer-tracklet-applemots-transfer MAX="0" DTYPE="fp32" CKPT=tracklet_applemots_transfer_ckpt:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/infer_tracklet.py \
        --config "{{tracklet_bft_official_schedulefree_config}}" --checkpoint "{{CKPT}}" --use-ema \
        --image-dir "{{tracklet_seq_images}}" --output-json "{{tracklet_applemots_transfer_infer_json}}" \
        --output-mot "outputs/tracklet_pseudomot_retrack_optuna_applemots_bft_schedulefree_smoke/infer_30fps/tracks_mot.txt" \
        --max-frames "{{MAX}}" --dtype "{{DTYPE}}"

# Render AppleMOTS-pretrained tomato transfer inference JSON -> mp4 only.
video-tracklet-applemots-transfer FPS="3" MAX="0":
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/visualize_tracks.py \
        --tracks-json "{{tracklet_applemots_transfer_infer_json}}" --image-dir "{{tracklet_seq_images}}" \
        --output-dir "" \
        --output-video "outputs/tracklet_pseudomot_retrack_optuna_applemots_bft_schedulefree_smoke/infer_30fps/tracks.mp4" \
        --fps "{{FPS}}" --show-score --max-frames "{{MAX}}"

# Compare Optuna re-tracked pseudo labels and AppleMOTS-pretrained tomato transfer output.
compare-tracklet-applemots-transfer:
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/compare_track_json.py \
        --old-json "{{tracklet_5fps_optuna_retrack_json}}" --new-json "{{tracklet_applemots_transfer_infer_json}}" \
        --output-json "{{tracklet_applemots_transfer_comparison_json}}"

# Re-track frame-level MOTIP bbox detections with a ByteTrack-style IoU tracker.
# Defaults are tuned for the tomato sequence; outputs are JSON + MOTChallenge txt + summary.
retrack-tracklet-5fps TRACK="0.80" LOW="0.20" NEW="0.95" MATCH="0.10" LOW_MATCH="0.10" MAX_AGE="60" NMS="0.70" VEL="1.0":
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/retrack_detections.py \
        --input-json "{{tracklet_5fps_infer_json}}" \
        --output-json "{{tracklet_5fps_retrack_json}}" \
        --output-mot "{{tracklet_5fps_retrack_mot}}" \
        --summary-json "{{tracklet_5fps_retrack_summary}}" \
        --track-thresh "{{TRACK}}" --low-thresh "{{LOW}}" --new-track-thresh "{{NEW}}" \
        --match-thresh "{{MATCH}}" --low-match-thresh "{{LOW_MATCH}}" \
        --max-age "{{MAX_AGE}}" --nms-iou "{{NMS}}" --velocity-weight "{{VEL}}"

# Render re-tracked bbox IDs -> mp4 only. FPS default 3, MAX=0 = all.
video-retrack-tracklet-5fps FPS="3" MAX="0":
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/visualize_tracks.py \
        --tracks-json "{{tracklet_5fps_retrack_json}}" --image-dir "{{tracklet_seq_images}}" \
        --output-dir "" \
        --output-video "outputs/tracklet_pseudomot_5fps_id16/retrack_bytetrack/tracks.mp4" \
        --fps "{{FPS}}" --show-score --max-frames "{{MAX}}"

# Tune re-tracking parameters with Optuna to favor longer tracklets while avoiding over-merge/dropout.
optuna-retrack-tracklet-5fps TRIALS="80" TIMEOUT="240":
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/tune_retrack_detections.py \
        --input-json "{{tracklet_5fps_infer_json}}" \
        --study-json "{{tracklet_5fps_optuna_study}}" \
        --best-output-json "{{tracklet_5fps_optuna_retrack_json}}" \
        --best-output-mot "{{tracklet_5fps_optuna_retrack_mot}}" \
        --best-summary-json "{{tracklet_5fps_optuna_retrack_summary}}" \
        --n-trials "{{TRIALS}}" --timeout "{{TIMEOUT}}"

# Render Optuna-tuned re-tracked bbox IDs -> mp4 only. FPS default 3, MAX=0 = all.
video-optuna-retrack-tracklet-5fps FPS="3" MAX="0":
    PYTHONPATH=. UV_PROJECT_ENVIRONMENT="{{venv}}" uv run --no-sync python tools/visualize_tracks.py \
        --tracks-json "{{tracklet_5fps_optuna_retrack_json}}" --image-dir "{{tracklet_seq_images}}" \
        --output-dir "" \
        --output-video "outputs/tracklet_pseudomot_5fps_id16/retrack_optuna/tracks.mp4" \
        --fps "{{FPS}}" --show-score --max-frames "{{MAX}}"
