
# T3.7 — Multi-Script Handwritten Indic Character Recognition

A two-stage CNN pipeline for recognising handwritten characters across four Indic scripts: **Devanagari**, **Tamil**, **Bengali**, and **Telugu**.

```
Input image → ScriptRouter (4-class CNN) → ScriptCNN (per-script classifier) → character label
```

---

## Architecture

### Stage 1 — Script Router
A lightweight 4-class CNN (`ScriptRouter`) identifies the script of the input image before passing it to the appropriate character classifier.

| Label | Script | Classes |
|-------|--------|---------|
| 0 | Devanagari | 46 |
| 1 | Tamil | 156 |
| 2 | Bengali | 84 |
| 3 | Telugu | 6 |

### Stage 2 — Per-Script Character CNN
`ScriptCNN` is a configurable-depth CNN (2–4 blocks). Each block follows `Conv → BN → ReLU → MaxPool`. A proportional FC head (`max(256, fc_in // 8)`) adapts to the number of classes per script.

| Script | Layers | Dropout | Batch |
|--------|--------|---------|-------|
| Devanagari | 4 | 0.5 | 256 |
| Tamil | 4 | 0.5 | 256 |
| Bengali | 4 | 0.5 | 256 |
| Telugu | 3 | 0.6 | 32 |

---

## Project Structure

```
A3/
├── train.ipynb           # Main training notebook (run on Kaggle or locally)
├── train_smoke.ipynb     # Smoke-test notebook for Kaggle push validation
├── smoke_test.py         # Local end-to-end smoke test script
├── requirements.txt      # Python dependencies (no torch/torchvision — pre-installed on Kaggle)
├── kernel-metadata.json  # Kaggle kernel push config
├── src/
│   ├── config.py         # Environment detection + all hyperparameters
│   ├── transforms.py     # Train/val image augmentation pipelines
│   ├── datasets.py       # ScriptImageDataset, BanglaLekhaDataset, TeluguCSVDataset, router loader
│   ├── models.py         # ScriptRouter, ScriptCNN, TwoStagePipeline
│   ├── trainer.py        # Training loop, checkpointing, CSV logging, curve plotting
│   ├── evaluate.py       # Per-script metrics, robustness, latency benchmark
│   ├── ablations.py      # 7 ablation studies on Devanagari
│   └── utils.py          # Seed, memory, model-wrapping helpers
├── data/
│   └── raw/              # Downloaded datasets (not committed)
│       ├── devanagari/
│       ├── tamil/
│       ├── bengali/
│       └── telugu/
└── working/              # Runtime output (checkpoints, logs, figures, results)
    ├── checkpoints/
    ├── logs/
    ├── figures/
    └── results/
```

---

## Datasets

| Script | Source | Kaggle Slug | Classes | ~Size |
|--------|--------|-------------|---------|-------|
| Devanagari | UCI / Kaggle | `medahmedkrichen/devanagari-handwritten-character-datase` | 46 | 92 k |
| Tamil | UTHCD / Kaggle | `faizalhajamohideen/uthcdtamil-handwritten-database` | 156 | ~500 k |
| Bengali | BanglaLekha-Isolated / Kaggle | `asefjamilajwad2/banglalekha-isolated` | 84 | ~166 k |
| Telugu | 6-Vowel CSV / Kaggle | `syamkakarla/telugu-6-vowel-dataset` | 6 | ~3 k |

---

## Quickstart

### Prerequisites

```bash
pip install -r requirements.txt
# torch and torchvision must be installed separately:
pip install torch torchvision
```

### Download datasets locally

```bash
kaggle datasets download medahmedkrichen/devanagari-handwritten-character-datase \
    -p data/raw/devanagari --unzip
kaggle datasets download faizalhajamohideen/uthcdtamil-handwritten-database \
    -p data/raw/tamil --unzip
kaggle datasets download asefjamilajwad2/banglalekha-isolated \
    -p data/raw/bengali --unzip
kaggle datasets download syamkakarla/telugu-6-vowel-dataset \
    -p data/raw/telugu --unzip
```

### Local smoke test (fast, ~5–10 min on CPU/MPS)

```bash
# Quick Python script smoke test
python smoke_test.py

# Or run the full notebook in smoke mode
KAGGLE_SMOKE=1 jupyter nbconvert --to notebook --execute train.ipynb \
    --output working/train_smoke_out.ipynb --ExecutePreprocessor.timeout=600
```

### Full local training

Open `train.ipynb` and set `KAGGLE_SMOKE_RUN = False` in cell 3, then run all cells.

---

## Kaggle Training

### Setup

1. Go to your Kaggle kernel and add these datasets via **+ Add Data**:
   - `medahmedkrichen/devanagari-handwritten-character-datase`
   - `faizalhajamohideen/uthcdtamil-handwritten-database`
   - `asefjamilajwad2/banglalekha-isolated`
   - `syamkakarla/telugu-6-vowel-dataset`
   - `dhruv10050/indic-handwriting-src` (source files)

2. Enable **GPU P100** or **2× T4** accelerator.

### Push via API

```bash
# Smoke run (KAGGLE_SMOKE_RUN = True in the notebook)
kaggle kernels push -p .

# Monitor
python3 -c "
import requests
TOKEN = '<your-kgat-token>'
r = requests.get(
    'https://www.kaggle.com/api/v1/kernels/status?username=dhruv10050&kernelslug=t3-7-multi-script-indic-handwriting-recognition',
    headers={'Authorization': f'Bearer {TOKEN}'}
)
print(r.json())
"
```

---

## Training Configuration

| Parameter | Smoke / Local | Full Run |
|-----------|--------------|----------|
| `IMG_SIZE` | 64 | 64 |
| `ROUTER_EPOCHS` | 3 | 15 |
| `CHAR_EPOCHS` | 3 | 25 |
| `ABLATION_EPOCHS` | 2 | 10 |
| `PATIENCE` | 2 | 5 |
| `CHAR_MAX` (imgs/class) | 50 | None (all) |
| `ROUTER_MAX` (imgs/script) | 200 | 5 000 |
| Optimizer | Adam | Adam |
| LR | 3e-4 | 3e-4 |
| Weight decay | 1e-4 | 1e-4 |
| LR schedule | CosineAnnealingLR | CosineAnnealingLR |
| AMP | CUDA only | CUDA only |

---

## Augmentation

**Train** — `Resize(64) → RandomCrop(64, pad=4) → RandomRotation(±10°) → RandomHorizontalFlip → ToTensor → Normalize → RandomErasing`

**Val / Test** — `Resize(64) → CenterCrop(64) → ToTensor → Normalize`

---

## Ablation Studies (Devanagari, 46 classes)

All 7 studies are run on Devanagari because it has the largest character inventory, making differences more statistically meaningful.

| # | Factor | Variants |
|---|--------|----------|
| 1 | CNN depth | 2 / 3 / **4** layers |
| 2 | Input resolution | 28 / 32 / **64** |
| 3 | BatchNorm | none / **with** |
| 4 | Dropout | 0.0 / 0.3 / **0.5** / 0.7 |
| 5 | Augmentation | none / **full** |
| 6 | LR scheduler | none / StepLR / **CosineAnn** / ReduceLROnPlateau |
| 7 | Architecture depth (proxy) | shallow 2-layer / **deep 4-layer** |

Bold = default configuration used in the main pipeline.

---

## Evaluation

- **Top-1 / Top-5 accuracy** on held-out test split
- **Macro F1 / Weighted F1** (via scikit-learn)
- **Confusion matrices** saved to `working/figures/`
- **Robustness evaluation** — 7 perturbation types (blur, rotation, brightness, crop) — full run only
- **Latency benchmark** — CPU single-image forward pass, 200 runs, reports router + classifier + end-to-end cost

---

## Streamlit App & Pretrained Models

A live interactive demo is available at the app repo. Run it locally:

```bash
git clone https://github.com/dhruv10050/T3.7-Multi-Script-Router-App
cd T3.7-Multi-Script-Router-App
pip install -r requirements.txt
streamlit run app.py
```

Checkpoints are loaded automatically from HuggingFace Hub on first launch:

| File | Script | Size |
|------|--------|------|
| `router_best.pth` | ScriptRouter | 26 MB |
| `devanagari_best.pth` | Devanagari (46 classes) | 30 MB |
| `tamil_best.pth` | Tamil (156 classes) | 31 MB |
| `bengali_best.pth` | Bengali (84 classes) | 30 MB |
| `telugu_best.pth` | Telugu (6 classes) | 102 MB |

Model hub: **[huggingface.co/dhruv10050/t3-7-indic-recognition](https://huggingface.co/dhruv10050/t3-7-indic-recognition)**

---

## Outputs

All outputs are written to `working/` (mapped to `/kaggle/working/` on Kaggle):

| Path | Content |
|------|---------|
| `working/checkpoints/<script>_best.pth` | Best checkpoint per script |
| `working/logs/<script>.csv` | Per-epoch loss/accuracy/LR log |
| `working/figures/curve_<script>.png` | Training curve plots |
| `working/figures/cm_<script>.png` | Confusion matrices |
| `working/figures/ablations.png` | Ablation bar charts |
| `working/figures/robustness.png` | Robustness accuracy chart |
| `working/results/final_results.json` | All metrics in JSON |
