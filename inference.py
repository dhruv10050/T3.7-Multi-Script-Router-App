"""Model loading and inference for the Streamlit app.

Loads the two-stage pipeline (ScriptRouter + ScriptCNN classifiers) from a
checkpoint directory.  Falls back gracefully when checkpoints are absent so
the UI can render a clear 'no model' state instead of crashing.
"""

import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

# Make src/ importable from both local runs and HuggingFace Spaces
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models import ScriptCNN, ScriptRouter, TwoStagePipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (must match training config)
# ---------------------------------------------------------------------------
IMG_SIZE = 64
DEVICE = torch.device("cpu")  # Spaces free tier; GPU optional

SCRIPT_CONFIGS: dict = {
    "devanagari": {"num_classes": 46, "num_layers": 4, "dropout": 0.5},
    "tamil":      {"num_classes": 156, "num_layers": 4, "dropout": 0.5},
    "bengali":    {"num_classes": 84,  "num_layers": 4, "dropout": 0.5},
    "telugu":     {"num_classes": 6,   "num_layers": 3, "dropout": 0.6},
}

VAL_TRANSFORM = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
])


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_state(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    # Strip DataParallel 'module.' prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# Default public model repo — override with HF_MODEL_REPO env var
_DEFAULT_HF_REPO = "dhruv10050/t3-7-indic-recognition"


def _try_hf_download(ckpt_dir: Path) -> None:
    """Download checkpoints from HuggingFace Hub.

    Uses HF_MODEL_REPO env var if set, otherwise falls back to the public
    default repo dhruv10050/t3-7-indic-recognition.
    """
    repo = os.environ.get("HF_MODEL_REPO", _DEFAULT_HF_REPO).strip()
    if not repo:
        return
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        files = list(list_repo_files(repo))
        for fname in files:
            if fname.endswith(".pth"):
                dest = ckpt_dir / Path(fname).name
                if not dest.exists():
                    local = hf_hub_download(repo_id=repo, filename=fname)
                    import shutil
                    shutil.copy(local, dest)
        print(f"[inference] Downloaded checkpoints from {repo}")
    except Exception as exc:
        print(f"[inference] HF Hub download skipped: {exc}")


def load_pipeline(ckpt_dir: str | Path = "checkpoints") -> tuple:
    """Load the TwoStagePipeline from *ckpt_dir*.

    Returns
    -------
    pipeline : TwoStagePipeline or None
        None if no router checkpoint is present.
    loaded_scripts : list[str]
        Which per-script classifiers were successfully loaded.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Try to pull from HuggingFace Hub if env var is set
    _try_hf_download(ckpt_dir)

    router_path = ckpt_dir / "router_best.pth"
    if not router_path.exists():
        return None, []

    router = _load_state(ScriptRouter(num_classes=4), router_path)

    classifiers, loaded = {}, []
    for script, cfg in SCRIPT_CONFIGS.items():
        p = ckpt_dir / f"{script}_best.pth"
        if p.exists():
            m = ScriptCNN(cfg["num_classes"], cfg["num_layers"], cfg["dropout"])
            classifiers[script] = _load_state(m, p)
            loaded.append(script)

    if not classifiers:
        return None, []

    pipeline = TwoStagePipeline(router, classifiers)
    pipeline.eval()
    return pipeline, loaded


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """Convert a PIL image (any mode/size) to a [1, 1, 64, 64] float tensor."""
    # Convert to grayscale
    img = pil_img.convert("L")
    tensor = VAL_TRANSFORM(img)     # [1, H, W]
    return tensor.unsqueeze(0)      # [1, 1, H, W]


def predict(pipeline: TwoStagePipeline, img_tensor: torch.Tensor) -> dict:
    """Run the two-stage pipeline and return a results dict."""
    with torch.no_grad():
        return pipeline(img_tensor, topk=5)
