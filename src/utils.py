import gc
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import DEVICE, NUM_GPUS, SEED


def seed_everything(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    # Deterministic cuDNN mode disables non-deterministic kernels to ensure
    # reproducible results across runs at a small throughput cost.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def wrap_model(model: nn.Module) -> nn.Module:
    model = model.to(DEVICE)
    # DataParallel splits each mini-batch across all available GPUs and
    # merges gradients automatically; it activates only on multi-GPU
    # environments such as a Kaggle 2×T4 instance.
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
    return model


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def find_path(*candidates) -> Optional[Path]:
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None
