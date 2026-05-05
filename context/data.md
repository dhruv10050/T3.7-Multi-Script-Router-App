# Data Plan

## Datasets Summary

| Script | Dataset | Source | Classes | Total Images | Format |
|--------|---------|--------|---------|-------------|--------|
| Devanagari | UCI Devanagari Handwritten Character Dataset (id 389) | [UCI ML Repo](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset) | 46 (digits + consonants) | ~92,000 | 32×32 PNG, grayscale |
| Tamil | uTHCD-Unconstrained Tamil Handwritten Database | [Kaggle: faizalhajamohideen/uthcdtamil-handwritten-database](https://www.kaggle.com/datasets/faizalhajamohideen/uthcdtamil-handwritten-database) | 156 | ~728,000 (all splits) | variable size PNG |
| Bengali | BanglaLekha-Isolated | [Mendeley: hf6sf8zrkc/2](https://data.mendeley.com/datasets/hf6sf8zrkc/2) | 84 | ~166,000 | variable size PNG |
| Telugu | Telugu 6-Vowel Dataset | [Kaggle: syamkakarla/telugu-6-vowel-dataset](https://www.kaggle.com/datasets/syamkakarla/telugu-6-vowel-dataset) | 6 (vowels) | ~1,200 | CSV (pixel values, 785 columns) |

---

## Download Instructions

### 1. UCI Devanagari (Devanagari)
```bash
# Download from UCI ML Repository (dataset ID 389)
pip install ucimlrepo
python -c "
from ucimlrepo import fetch_ucirepo
devanagari = fetch_ucirepo(id=389)
"
# Or direct download:
# https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset
```
Structure after download:
```
DevanagariHandwrittenCharacterDataset/
├── Train/   (36 consonants + 10 digits = 46 classes, ~78k images)
└── Test/    (~14k images)
```

### 2. uTHCD Tamil
```bash
kaggle datasets download -d faizalhajamohideen/uthcdtamil-handwritten-database
unzip uthcdtamil-handwritten-database.zip -d data/raw/tamil/
```
The dataset includes 4 pre-split variants (70-30, 80-20, 85-15, 90-10). Use the **85-15 split** (`uTHCD_c`) to stay consistent with other datasets.
Structure after download:
```
uTHCD_c(85-15-split)/
├── train/   (156 class folders)
└── test/    (156 class folders)
```

### 3. BanglaLekha-Isolated (Bengali)
```bash
# Download from Mendeley Data
# URL: https://data.mendeley.com/datasets/hf6sf8zrkc/2
# Manual download → data/raw/bengali/
```
Structure after download:
```
BanglaLekha-Isolated/
└── Images/   (84 class folders, ~166k images)
```

### 4. Telugu 6-Vowel Dataset
```bash
kaggle datasets download -d syamkakarla/telugu-6-vowel-dataset
unzip telugu-6-vowel-dataset.zip -d data/raw/telugu/
```
The dataset is in **CSV format** (not image files). Each row is one sample; the first 784 columns are pixel values (28×28 flattened) and the last column is the class label.
```python
import pandas as pd
df = pd.read_csv('data/raw/telugu/CSV_datasetsix_vowel_dataset_with_class.csv')
# Reshape: pixels = df.iloc[:, :784].values.reshape(-1, 28, 28)
# Labels  = df.iloc[:, 784].values  → 6 vowel classes
```
Class labels map to Telugu vowels: అ, ఆ, ఇ, ఈ, ఉ, ఊ (or whichever 6 are in the dataset — verify from CSV).

---

## Directory Structure After Download

```
A3/data/
├── raw/
│   ├── devanagari/     # original dataset folders
│   ├── tamil/
│   ├── bengali/
│   └── telugu/         # CSV file(s)
└── processed/
    ├── devanagari/     # 64×64 grayscale PNGs, split into train/val/test
    ├── tamil/
    ├── bengali/
    ├── telugu/
    └── router/         # 4-class script dataset (5k images per script)
        ├── train/
        │   ├── devanagari/
        │   ├── tamil/
        │   ├── bengali/
        │   └── telugu/
        ├── val/
        └── test/
```

---

## Preprocessing Steps

Applied to ALL images before saving to `data/processed/`:

1. **Load as grayscale** (convert RGB if necessary)
2. **Invert if needed**: dataset-specific — Devanagari uses white background (keep as-is); Tamil/Bengali may vary (check mean pixel intensity); Telugu is reconstructed from CSV pixel values (verify orientation)
   - Telugu special step: reshape row of 784 values → 28×28 numpy array → convert to PIL Image → then proceed with crop/resize
3. **Crop to bounding box**: remove blank border pixels, add 10% padding on each side
4. **Resize to 64×64**: bilinear interpolation
5. **Save as uint8 PNG** (no normalization at storage stage — normalize at training time)

### Normalization (at training time via transforms)
```python
transforms.Normalize(mean=[0.5], std=[0.5])
# Converts [0, 1] float → [-1, 1]
```

---

## Train / Val / Test Split

| Split | Fraction | Stratification |
|-------|----------|---------------|
| Train | 70% | Per-class stratified |
| Val | 15% | Per-class stratified |
| Test | 15% | Per-class stratified (held out, never used for model selection) |

Use `sklearn.model_selection.train_test_split` with `stratify=labels`.

Fixed random seed: **42** for reproducibility.

---

## Router Dataset Construction

The script router needs a balanced 4-class dataset:

| Script | Images sampled | Note |
|--------|---------------|------|
| Devanagari | 5,000 | stratified across 46 classes |
| Tamil | 5,000 | stratified across 156 classes |
| Bengali | 5,000 | stratified across 84 classes |
| Telugu | 1,200 (all samples) | only ~200/class available — use all |
| **Total** | **~16,200** | Telugu is the smallest script |

> **Note on Telugu imbalance:** Because Telugu has only ~1,200 total samples vs 5,000 for others, apply class weighting in the router loss (`nn.CrossEntropyLoss(weight=...)`) and oversample Telugu during router training using `WeightedRandomSampler`.

Sampling strategy: take `ceil(5000 / num_classes)` images per class, shuffle, truncate to 5000.

---

## Class Imbalance Analysis

- **Devanagari (46 classes):** ~2,000 images/class → well balanced
- **Tamil (156 classes):** ~4,700 images/class (728k total across all splits) → use 85-15 split → well balanced
- **Bengali (84 classes):** ~2,000 images/class → well balanced
- **Telugu (6 classes):** ~200 images/class → **very small dataset**, high augmentation required

**Strategy for Telugu (tiny dataset ~1,200 samples):**
- Apply aggressive augmentation (rotation ±15°, elastic distortion, brightness jitter)
- Use weighted random sampler: `torch.utils.data.WeightedRandomSampler`
- Class weights in loss function: `nn.CrossEntropyLoss(weight=class_weights)`
- Consider k-fold cross-validation (5-fold) for reliable evaluation given small size

**Strategy for Tamil (minority router representation):**
- Apply heavier augmentation (rotation ±15°, stronger affine transforms) during router training

---

## Data Augmentation Policy

| Transform | Train | Val/Test |
|-----------|-------|---------|
| RandomRotation(±10°) | ✓ | ✗ |
| RandomAffine(translate=0.1, shear=5°) | ✓ | ✗ |
| RandomErasing(p=0.1) | ✓ | ✗ |
| GaussianNoise(std=0.05) | ✓ | ✗ |
| Resize(64×64) | ✓ | ✓ |
| ToTensor + Normalize(0.5, 0.5) | ✓ | ✓ |

---

## Unicode Mapping

Each predicted class index maps to a Unicode character for display in the app.

| Script | Unicode Block | Range |
|--------|--------------|-------|
| Devanagari | Devanagari | U+0900–U+097F |
| Tamil | Tamil | U+0B80–U+0BFF |
| Bengali | Bengali | U+0980–U+09FF |
| Telugu | Telugu | U+0C00–U+0C7F |

Unicode maps will be stored in `src/utils/unicode_maps.py` as Python dicts:
```python
DEVANAGARI_MAP = {0: 'क', 1: 'ख', 2: 'ग', ...}  # 46 entries
TAMIL_MAP      = {0: 'அ', 1: 'ஆ', ...}            # 156 entries
BENGALI_MAP    = {0: 'অ', 1: 'আ', ...}             # 84 entries
TELUGU_MAP     = {0: 'అ', 1: 'ఆ', 2: 'ఇ', 3: 'ఈ', 4: 'ఉ', 5: 'ఊ'}  # 6 vowels
```
> Verify the exact 6 Telugu vowel classes from the CSV label column before finalising this map.

---

## Dataset Statistics to Report

Compute and include in the report:
- Images per class (min, max, mean) per script
- Total train/val/test counts per script
- Pixel intensity distribution (histogram) per script
- Sample grid visualization (3 samples × 5 classes) per script
- Image size distribution before resizing (for Tamil and Bengali)
- For Telugu: verify class label encoding from CSV, confirm 6 distinct vowel classes
