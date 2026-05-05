"""Upload the 5 trained model checkpoints to HuggingFace Hub.

Usage
-----
  export HF_TOKEN=hf_xxxxxxxxxxxx
  python3 upload_to_hf.py

The script creates (or reuses) a model repo at
  https://huggingface.co/<HF_USER>/t3-7-indic-recognition
and uploads the 5 .pth files from kaggle_output/checkpoints/.

After running this, set HF_MODEL_REPO=<HF_USER>/t3-7-indic-recognition
in your Streamlit Spaces environment variables and the app will
auto-download the checkpoints on first launch.
"""

import os
import sys
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
TOKEN    = os.environ.get("HF_TOKEN", "").strip()
HF_USER  = os.environ.get("HF_USER", "").strip()   # your HuggingFace username
REPO_NAME = "t3-7-indic-recognition"

CKPT_DIR = Path(__file__).parent / "kaggle_output" / "checkpoints"
FILES = [
    "router_best.pth",
    "devanagari_best.pth",
    "tamil_best.pth",
    "bengali_best.pth",
    "telugu_best.pth",
]

# ─────────────────────────────────────────────────────────────────────────────
if not TOKEN:
    print("ERROR: Set HF_TOKEN env var to your HuggingFace write token.")
    print("  Get one at https://huggingface.co/settings/tokens")
    sys.exit(1)

if not HF_USER:
    # try to resolve from the API
    try:
        from huggingface_hub import whoami
        info = whoami(token=TOKEN)
        HF_USER = info["name"]
        print(f"Logged in as: {HF_USER}")
    except Exception as e:
        print(f"ERROR: Set HF_USER env var (your HuggingFace username). ({e})")
        sys.exit(1)

from huggingface_hub import HfApi, create_repo

api = HfApi(token=TOKEN)
repo_id = f"{HF_USER}/{REPO_NAME}"

# Create repo if it doesn't exist
print(f"Creating/updating repo: {repo_id}")
create_repo(repo_id, token=TOKEN, repo_type="model", exist_ok=True, private=False)

# Upload model card
card = f"""---
language: []
tags:
  - pytorch
  - image-classification
  - indic-scripts
  - devanagari
  - tamil
  - bengali
  - telugu
license: mit
---

# T3.7 Multi-Script Indic Handwriting Recognition

Two-stage CNN pipeline: **ScriptRouter** (4-class) → **ScriptCNN** (per-script).

| Script | Classes | Top-1 | Top-5 | Macro F1 |
|---|---|---|---|---|
| Router | 4 | 99.92% | — | — |
| Devanagari | 46 | 99.48% | 99.99% | 99.41 |
| Tamil | 156 | 97.30% | 99.77% | 95.96 |
| Bengali | 84 | 93.02% | 98.79% | 93.12 |
| Telugu | 6 | 98.89% | 100.0% | 98.88 |

E2E CPU latency: **10.21 ms**.  Trained from scratch, no transfer learning.

## Usage
```python
from inference import load_pipeline, preprocess, predict
import os
os.environ["HF_MODEL_REPO"] = "{repo_id}"
pipeline, scripts = load_pipeline("./checkpoints")
```

## Files
- `router_best.pth` — ScriptRouter (4-class CNN)
- `devanagari_best.pth` — Devanagari classifier (46 classes)
- `tamil_best.pth` — Tamil classifier (156 classes)
- `bengali_best.pth` — Bengali classifier (84 classes)
- `telugu_best.pth` — Telugu classifier (6 vowel classes)
"""

api.upload_file(
    path_or_fileobj=card.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Add model card",
)
print("Model card uploaded.")

# Upload checkpoints
for fname in FILES:
    src = CKPT_DIR / fname
    if not src.exists():
        print(f"  SKIP (not found): {src}")
        continue
    size_mb = src.stat().st_size / 1e6
    print(f"  Uploading {fname} ({size_mb:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add {fname}",
    )
    print(f"  ✓ {fname}")

print(f"\nDone. Model available at: https://huggingface.co/{repo_id}")
print(f"\nTo use in Streamlit Spaces, add environment variable:")
print(f"  HF_MODEL_REPO = {repo_id}")
