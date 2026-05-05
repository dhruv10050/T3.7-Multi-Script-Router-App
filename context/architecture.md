# Architecture Plan

## Overview: Two-Stage Pipeline

```
Input Image (canvas draw or file upload)
             │
             ▼
   ┌─────────────────────┐
   │   Preprocessing     │  resize → 64×64 grayscale, invert, normalize
   └─────────────────────┘
             │
             ▼
   ┌─────────────────────┐
   │   Model 1           │  4-class output:
   │   Script Router     │  Devanagari / Tamil / Bengali / Telugu
   │   (~100k params)    │
   └─────────────────────┘
             │  predicted script
             ▼
   ┌─────────────────────┐
   │   Model 2           │  N-class output (script-specific):
   │   Character CNN     │  46 / 156 / 84 / 6 classes
   │   (~2.8M params)    │
   └─────────────────────┘
             │
             ▼
   Unicode prediction + top-5 confidences
```

---

## Model 1: Script Router

**Task:** 4-class classification — Devanagari | Tamil | Bengali | Telugu  
**Input:** 1 × 64 × 64 (grayscale, normalized)  
**Output:** 4-dim softmax  
**Target params:** ~100k

### Layer Table

| # | Layer | Config | Output Shape |
|---|-------|--------|--------------|
| 1 | Conv2d | 1→32, 3×3, pad=1 | 32×64×64 |
| 2 | BatchNorm2d + ReLU | — | 32×64×64 |
| 3 | MaxPool2d | 2×2 | 32×32×32 |
| 4 | Conv2d | 32→64, 3×3, pad=1 | 64×32×32 |
| 5 | BatchNorm2d + ReLU | — | 64×32×32 |
| 6 | MaxPool2d | 2×2 | 64×16×16 |
| 7 | Conv2d | 64→128, 3×3, pad=1 | 128×16×16 |
| 8 | BatchNorm2d + ReLU | — | 128×16×16 |
| 9 | MaxPool2d | 2×2 | 128×8×8 |
| 10 | Flatten | — | 8192 |
| 11 | Linear + ReLU | 8192→256 | 256 |
| 12 | Dropout | p=0.4 | 256 |
| 13 | Linear + Softmax | 256→4 | 4 |

**Training strategy:**
- Sample 5,000 images per script (20k total, balanced)
- 70/15/15 train/val/test split
- Adam, lr=3e-4, weight_decay=1e-4
- CosineAnnealingLR (T_max=20)
- 20 epochs, batch_size=128

---

## Model 2: Character Classifiers (4 separate models)

**Design choice:** 4 independent per-script models rather than one shared model.  
**Rationale:** Avoids class-count imbalance across scripts, allows per-script tuning, cleaner inference path.

### Per-Script Class Counts

| Script | Dataset | Classes | Training images (approx.) |
|--------|---------|---------|--------------------------|
| Devanagari | UCI (full set) | 46 | ~64,000 |
| Tamil | uTHCD (85-15 split) | 156 | ~618,000 |
| Bengali | BanglaLekha-Isolated | 84 | ~116,000 |
| Telugu | Telugu 6-Vowel Dataset | 6 | ~840 (70% of ~1,200) |

### ScriptCNN Architecture (shared template, instantiated per script)

| # | Layer | Config | Output Shape |
|---|-------|--------|--------------|
| 1 | Conv2d | 1→32, 3×3, pad=1 | 32×64×64 |
| 2 | BatchNorm2d + ReLU | — | 32×64×64 |
| 3 | MaxPool2d | 2×2 | 32×32×32 |
| 4 | Conv2d | 32→64, 3×3, pad=1 | 64×32×32 |
| 5 | BatchNorm2d + ReLU | — | 64×32×32 |
| 6 | MaxPool2d | 2×2 | 64×16×16 |
| 7 | Conv2d | 64→128, 3×3, pad=1 | 128×16×16 |
| 8 | BatchNorm2d + ReLU | — | 128×16×16 |
| 9 | MaxPool2d | 2×2 | 128×8×8 |
| 10 | Conv2d | 128→256, 3×3, pad=1 | 256×8×8 |
| 11 | BatchNorm2d + ReLU | — | 256×8×8 |
| 12 | MaxPool2d | 2×2 | 256×4×4 |
| 13 | Flatten | — | 4096 |
| 14 | Linear + ReLU | 4096→512 | 512 |
| 15 | Dropout | p=0.5 | 512 |
| 16 | Linear + Softmax | 512→num_classes | num_classes |

**Training strategy (per-script):**
- Adam, lr=3e-4, weight_decay=1e-4
- CosineAnnealingLR (T_max=30)
- 30 epochs, batch_size=64
- Early stopping: patience=5 on val accuracy
- Save best checkpoint by val accuracy

---

## Input Preprocessing Pipeline

```
Raw canvas image (RGBA, 280×280)
    │
    ├─ Convert to grayscale
    │
    ├─ Invert if white background (canvas draws white stroke on black bg)
    │     threshold: if mean pixel < 128 → invert
    │
    ├─ Crop to tight bounding box of non-zero pixels (with 10% padding)
    │
    ├─ Resize to 64×64 (bilinear)
    │
    ├─ Convert to float32, scale to [0, 1]
    │
    └─ Normalize: mean=0.5, std=0.5  → range [-1, 1]
```

### Training Augmentation (applied only during training)

| Transform | Parameters |
|-----------|-----------|
| RandomRotation | ±10° |
| RandomAffine | translate=(0.1, 0.1), shear=5° |
| RandomErasing | p=0.1, scale=(0.02, 0.1) |
| GaussianNoise | std=0.05 (custom transform) |
| No horizontal/vertical flip | (scripts are orientation-sensitive) |

---

## End-to-End Inference

```python
# Pseudocode
image = preprocess(raw_canvas_image)           # → Tensor[1, 1, 64, 64]
script_logits = router(image)                  # → Tensor[1, 4]
script_id = script_logits.argmax()             # 0=Deva, 1=Tamil, 2=Bengali, 3=Telugu
char_models = {"devanagari": ..., "tamil": ..., "bengali": ..., "telugu": ...}
char_logits = char_models[script_id](image)    # → Tensor[1, num_classes]
top5 = char_logits.topk(5)                     # top-5 predictions + confidences
unicode_chars = [script_unicode_map[script_id][i] for i in top5.indices]
```

---

## File Structure

```
A3/
├── src/
│   ├── models/
│   │   ├── script_router.py       # ScriptRouter CNN definition
│   │   ├── char_classifier.py     # ScriptCNN definition (shared template)
│   │   └── pipeline.py            # TwoStagePipeline wrapper
│   ├── data/
│   │   ├── datasets.py            # Dataset classes per script
│   │   └── transforms.py          # Preprocessing + augmentation
│   └── utils/
│       ├── unicode_maps.py        # Script → character → Unicode mappings
│       └── metrics.py             # Accuracy, F1, confusion matrix helpers
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_router.ipynb
│   ├── 03_train_classifiers.ipynb
│   └── 04_evaluation.ipynb
├── app/
│   ├── app.py                     # Streamlit app entry point
│   └── assets/                    # Practice-mode character images
├── checkpoints/                   # Saved model weights (.pth)
├── requirements.txt
└── README.md
```
