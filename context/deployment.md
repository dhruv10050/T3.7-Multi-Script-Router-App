# Deployment Plan

## Deliverables

1. **GitHub Repository** — committed and pushed at `git push origin main`
2. **Kaggle Kernel** — `dhruv10050/t3-7-multi-script-indic-handwriting-recognition` (private, GPU enabled)
3. **README.md** — project-level documentation at repo root

---

## Actual Deployment Architecture

Training and evaluation run entirely on **Kaggle** via a Jupyter notebook (`train.ipynb`) pushed through the Kaggle REST API.  There is no Streamlit/HuggingFace app component — all deliverables are Kaggle output artifacts.

### File Structure (actual)

```
A3/
├── train.ipynb              # Orchestration notebook (21 cells)
├── kernel-metadata.json     # Kaggle kernel config
├── requirements.txt         # Non-torch deps only (torch pre-installed on Kaggle)
├── README.md
├── src/
│   ├── config.py            # ENV detection, hyperparameters
│   ├── datasets.py          # Dataset classes + DataLoader factories
│   ├── models.py            # ScriptRouter, ScriptCNN
│   ├── trainer.py           # Training loop, AMP probe, checkpointing
│   ├── transforms.py        # Augmentation pipeline
│   ├── evaluate.py          # Per-script eval, robustness, latency
│   ├── ablations.py         # 7 ablation studies (Devanagari)
│   └── utils.py             # wrap_model, fmt_time, find_path
├── _push_dataset2.py        # Blob-upload src/ to Kaggle dataset
├── _push_kernel.py          # Push train.ipynb to Kaggle kernel
└── _monitor.py              # Poll kernel status + log stream

working/                     # Created at runtime (Kaggle /kaggle/working)
├── checkpoints/
│   ├── router_best.pth
│   ├── devanagari_best.pth
│   ├── tamil_best.pth
│   ├── bengali_best.pth
│   └── telugu_best.pth
├── logs/                    # CSV training logs per tag
└── figures/                 # Training curves, confusion matrices
```

---

## Kaggle Kernel Configuration

**Kernel ID:** `dhruv10050/t3-7-multi-script-indic-handwriting-recognition`

```json
{
  "id": "dhruv10050/t3-7-multi-script-indic-handwriting-recognition",
  "title": "T3.7 Multi-Script Indic Handwriting Recognition",
  "code_file": "train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "docker_image_pinning_type": "original",
  "dataset_sources": [
    "dhruv10050/indic-handwriting-src",
    "medahmedkrichen/devanagari-handwritten-character-datase",
    "faizalhajamohideen/uthcdtamil-handwritten-database",
    "asefjamilajwad2/banglalekha-isolated",
    "syamkakarla/telugu-6-vowel-dataset"
  ]
}
```

**Source dataset:** `dhruv10050/indic-handwriting-src` holds all `src/*.py` files.  
Updated via `_push_dataset2.py` (uses Kaggle blob upload API + Bearer token, **not** the CLI).

---

## Kaggle Push Workflow

```bash
# 1. Update src/ on Kaggle
python3 _push_dataset2.py

# 2. Push notebook kernel version
python3 _push_kernel.py          # returns {"versionNumber": N, ...}

# 3. Monitor execution
python3 _monitor.py              # polls status + log every 30s
```

**Important:** All Kaggle write API calls require `Authorization: Bearer KGAT_...` header.  
The `kaggle` CLI returns HTTP 401 for all write operations (push, upload) — use `requests` directly.

---

## Run-Mode Switch

Cell 3 of `train.ipynb` controls the run mode:

```python
KAGGLE_SMOKE_RUN = False   # True → smoke (3 epochs, 50 imgs/class, ~15 min)
                            # False → full run (15/25 epochs, all data, ~5–6 hrs)
```

Setting `KAGGLE_SMOKE_RUN = True` exports `KAGGLE_SMOKE=1` to the environment,  
which `src/config.py` reads at module-load time via `os.environ.get("KAGGLE_SMOKE", "0")`.

---

## CUDA Compatibility Fix (trainer.py)

Kaggle may assign a Tesla P100 (SM 6.0).  Later PyTorch builds (≥2.7, +cu128) dropped SM 6.0.  
Two safeguards are in place:

1. **Cell 2 (install cell)** — probes CUDA in a subprocess before any `import torch` in the kernel process.  If the probe fails, installs `torch<2.7+cu121` which restores SM 6.0 support.  
2. **`_amp_supported()` in trainer.py** — probes fp16 autocast + `cuda.synchronize()` at module load; sets `_USE_AMP = False` on failure so training falls back to fp32 without crashing.

---

## Checkpoint Outputs

| File | Contents |
|------|----------|
| `router_best.pth` | Best script router weights (4-class) |
| `devanagari_best.pth` | Best character classifier — 46 classes |
| `tamil_best.pth` | Best character classifier — 156 classes |
| `bengali_best.pth` | Best character classifier — 84 classes |
| `telugu_best.pth` | Best character classifier — 6 classes |

All checkpoints are saved with `torch.save({"epoch": ..., "state_dict": ..., "opt_state": ..., "val_acc": ...}, path)`.


### File Structure

```
app/
├── app.py                  # Main Streamlit app entry point
├── inference.py            # Preprocessing + model inference logic
├── unicode_display.py      # Unicode character rendering helpers
└── assets/
    └── practice_chars/     # Reference images for practice mode
        ├── devanagari/
        ├── tamil/
        ├── bengali/
        └── telugu/

checkpoints/
├── script_router.pth
├── devanagari_classifier.pth
├── tamil_classifier.pth
├── bengali_classifier.pth
└── telugu_classifier.pth
```

---

## App Features

### Core: Live Prediction Mode

```
┌─────────────────────────────────────────────────────┐
│  Handwritten Indic Script Recognition               │
├────────────────────────┬────────────────────────────┤
│  Draw a character here │  Predictions               │
│  ┌──────────────────┐  │  Script:  Bengali  (98.2%) │
│  │                  │  │  ─────────────────────────  │
│  │   [canvas]       │  │  Top 5 predictions:        │
│  │                  │  │  1. অ  (U+0985)  94.1%     │
│  │                  │  │  2. আ  (U+0986)   3.2%     │
│  └──────────────────┘  │  3. ও  (U+0993)   1.4%     │
│  [Clear] [Predict]     │  4. এ  (U+098F)   0.8%     │
│                        │  5. ই  (U+0987)   0.5%     │
│  — or —                │                            │
│  [Upload Image]        │  Confidence bar chart      │
└────────────────────────┴────────────────────────────┘
```

**Components:**
- `streamlit-drawable-canvas`: 280×280px, stroke width slider, black pen on white bg
- Script prediction shown as a colored badge
- Top-5 character predictions with Unicode codepoints and horizontal confidence bars
- "Copy character" button for each prediction

### Optional: Practice Mode

```
┌─────────────────────────────────────────────────────┐
│  Practice Mode                                      │
│  Script: [Devanagari ▼]                             │
│                                                     │
│  Target character: क (ka)                           │
│  ┌──────────┐   ┌──────────────┐                   │
│  │ Reference│   │  Your drawing│                    │
│  │   image  │   │  [canvas]    │                    │
│  └──────────┘   └──────────────┘                   │
│                                                     │
│  [Submit]  →  ✓ Correct! (96.2% confidence)        │
│               or  ✗ Got: ग — try again             │
│  [Next character]                                   │
└─────────────────────────────────────────────────────┘
```

---

## app.py Implementation Notes

```python
# Key imports
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="Indic Script Recognition", layout="wide")

# Cache model loading (only load once)
@st.cache_resource
def load_models():
    router = load_script_router("checkpoints/script_router.pth")
    classifiers = {
        "devanagari": load_char_classifier("checkpoints/devanagari_classifier.pth", 46),
        "tamil":      load_char_classifier("checkpoints/tamil_classifier.pth", 156),
        "bengali":    load_char_classifier("checkpoints/bengali_classifier.pth", 84),
        "telugu":     load_char_classifier("checkpoints/telugu_classifier.pth", 6),
    }
    return router, classifiers
```

**Key implementation decisions:**
- Use `@st.cache_resource` for model loading to avoid reloading on each interaction
- Inference runs on CPU (no CUDA on HuggingFace Spaces free tier)
- Canvas output is RGBA numpy array → convert to grayscale PIL image → preprocess
- Models stored in `checkpoints/` within the repo (use Git LFS for `.pth` files, or load from HuggingFace Hub)

---

## requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.32.0
streamlit-drawable-canvas>=0.9.3
Pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
```

---

## HuggingFace Spaces Deployment

### Steps

1. Create HuggingFace account at huggingface.co
2. Create new Space:
   - Space name: `indic-script-recognition`
   - SDK: **Streamlit**
   - Hardware: **CPU basic** (free tier)
   - Visibility: Public

3. Clone the Space repo:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/indic-script-recognition
   ```

4. Copy app files:
   ```bash
   cp app/app.py .
   cp app/inference.py .
   cp requirements.txt .
   ```

5. Handle model weights (choose one):
   - **Option A — Git LFS** (if total < 500MB):
     ```bash
     git lfs install
     git lfs track "*.pth"
     git add .gitattributes
     cp -r checkpoints/ .
     ```
   - **Option B — HuggingFace Hub** (recommended for large files):
     ```python
     from huggingface_hub import hf_hub_download
     # In app.py, load models from Hub at startup
     path = hf_hub_download(repo_id="YOUR_USERNAME/indic-models", filename="script_router.pth")
     ```

6. Add `README.md` with YAML frontmatter:
   ```yaml
   ---
   title: Indic Script Recognition
   emoji: ✍️
   colorFrom: blue
   colorTo: purple
   sdk: streamlit
   sdk_version: 1.32.0
   app_file: app.py
   pinned: false
   ---
   ```

7. Push to HuggingFace:
   ```bash
   git add . && git commit -m "Initial deployment" && git push
   ```

### Fallback: Streamlit Community Cloud

1. Push full repo to GitHub (public)
2. Go to share.streamlit.io → New app
3. Select repo, branch `main`, file `app/app.py`
4. Deploy (free, auto-rebuilds on push)

---

## GitHub Repository Structure

```
A3/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes          # Git LFS tracking for .pth files
├── app/
│   └── app.py
├── src/
│   ├── models/
│   ├── data/
│   └── utils/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_router.ipynb
│   ├── 03_train_classifiers.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_ablations.ipynb
├── checkpoints/            # tracked with Git LFS
└── results/                # figures + JSON metrics
```

### README.md Sections

1. **Project title + one-line description**
2. **Demo GIF / screenshot** (link to live Space)
3. **Datasets** (names + links)
4. **Architecture** (brief — pipeline diagram)
5. **Results table** (top-1 accuracy per script)
6. **Quick start** (clone → install → run app locally)
7. **Training** (link to Colab notebooks)
8. **Report** (link to PDF)
9. **License**

---

## One-Slide Pitch (LinkedIn PNG)

**Content for the single slide:**
- Title: "Multi-Script Handwritten Indic Recognition"
- Subtitle: "4 scripts × ~300 characters recognized in real-time"
- Architecture diagram (simplified, 3 boxes)
- Results table (clean, 4-row)
- Demo QR code (link to HuggingFace Space)
- Tech stack icons: PyTorch, Streamlit, HuggingFace

**Tool:** Canva or Figma → export as PNG (1920×1080 or 1080×1080)
