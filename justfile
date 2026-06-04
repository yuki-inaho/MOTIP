set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
venv := "/home/kasm-user/Desktop/MOTIP/.venv"

sync:
    UV_PROJECT_ENVIRONMENT="{{venv}}" uv sync

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
