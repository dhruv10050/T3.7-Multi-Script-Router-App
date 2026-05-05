import os
import gc
from pathlib import Path

import torch


def _detect_env() -> str:
    if os.path.exists("/kaggle/working"):         return "kaggle"
    if torch.backends.mps.is_available():         return "mac"
    if torch.cuda.is_available():                 return "gpu"
    return "cpu"

ENV: str = _detect_env()

if ENV == "kaggle":
    NUM_GPUS = torch.cuda.device_count()
    DEVICE   = torch.device("cuda")
elif ENV == "mac":
    DEVICE, NUM_GPUS = torch.device("mps"), 0
elif ENV == "gpu":
    DEVICE, NUM_GPUS = torch.device("cuda"), 1
else:
    DEVICE, NUM_GPUS = torch.device("cpu"), 0

TEST_RUN:     bool = ENV in ("mac", "cpu")
# KAGGLE_SMOKE must be set as an environment variable before this module is
# imported; it cannot be toggled at runtime once config is loaded.
KAGGLE_SMOKE: bool = os.environ.get("KAGGLE_SMOKE", "0") == "1"
_SMALL_RUN:   bool = TEST_RUN or KAGGLE_SMOKE
RUN_ABLATIONS: bool = not _SMALL_RUN

SEED:         int   = 42
IMG_SIZE:     int   = 64
LR:           float = 3e-4
WEIGHT_DECAY: float = 1e-4

ROUTER_BATCH:  int = 512 if ENV == "kaggle" else 32
CHAR_BATCH:    int = 256 if ENV == "kaggle" else 32
TELUGU_BATCH:  int = 32
NUM_WORKERS:   int = 4   if ENV == "kaggle" else 0
PIN_MEMORY:    bool = DEVICE.type == "cuda"
PERSISTENT_W:  bool = NUM_WORKERS > 0

# Reduced epoch counts keep smoke and local test runs fast; full-run
# values target approximately 5 hours of training on a Kaggle P100 GPU.
ROUTER_EPOCHS:   int = 3  if _SMALL_RUN else 15
CHAR_EPOCHS:     int = 3  if _SMALL_RUN else 25
ABLATION_EPOCHS: int = 2  if _SMALL_RUN else 10
PATIENCE:        int = 2  if _SMALL_RUN else 5
ROUTER_MAX:      int = 200  if _SMALL_RUN else 5000
# Setting CHAR_MAX to None during a full run removes the per-class sample
# cap so the classifier trains on the entire available dataset.
CHAR_MAX:        int = 50   if _SMALL_RUN else None

NORM_MEAN = [0.5]
NORM_STD  = [0.5]

WORK_DIR: Path = Path("/kaggle/working") if ENV == "kaggle" else Path("./working")
CKPT_DIR: Path = WORK_DIR / "checkpoints"
LOG_DIR:  Path = WORK_DIR / "logs"
RES_DIR:  Path = WORK_DIR / "results"
FIG_DIR:  Path = WORK_DIR / "figures"

for _d in [CKPT_DIR, LOG_DIR, RES_DIR, FIG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def print_config() -> None:
    gpu_info = ""
    if ENV == "kaggle":
        for i in range(NUM_GPUS):
            p = torch.cuda.get_device_properties(i)
            gpu_info += f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB\n"
    elif ENV == "mac":
        gpu_info = "  Apple Silicon MPS\n"
    elif ENV == "gpu":
        gpu_info = f"  CUDA: {torch.cuda.get_device_name(0)}\n"
    else:
        gpu_info = "  CPU only\n"

    print(
        f"ENV={ENV}  DEVICE={DEVICE}  GPUs={NUM_GPUS}\n"
        f"{gpu_info}"
        f"TEST_RUN={TEST_RUN}  KAGGLE_SMOKE={KAGGLE_SMOKE}  RUN_ABLATIONS={RUN_ABLATIONS}\n"
        f"IMG_SIZE={IMG_SIZE}  BATCH(router/char)={ROUTER_BATCH}/{CHAR_BATCH}\n"
        f"EPOCHS(router/char/ablation)={ROUTER_EPOCHS}/{CHAR_EPOCHS}/{ABLATION_EPOCHS}\n"
        f"CHAR_MAX={CHAR_MAX}  ROUTER_MAX={ROUTER_MAX}\n"
        f"WORK_DIR={WORK_DIR}"
    )
