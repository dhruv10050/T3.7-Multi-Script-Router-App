# Report Plan (ICVGIP Style)

## Format Requirements

- **Length:** 6–8 pages (ICVGIP two-column format)
- **Style:** ICVGIP 2024 template (LaTeX)
- **Template:** [ICVGIP LaTeX template](https://cvit.iiit.ac.in/icvgip/) — use `\documentclass{article}` with two-column layout
- **Output:** PDF submitted with assignment
- **Figures:** Inline, numbered, referenced in text
- **Tables:** Numbered, with captions above
- **References:** IEEE style, 10–15 citations minimum

---

## Section Outline

### 1. Title + Abstract (0.25 pages)

**Title:** `Multi-Script Handwritten Indic Character Recognition via a Two-Stage CNN Pipeline`

**Abstract (~150 words):**
- Problem: handwritten recognition for 4 Indic scripts simultaneously
- Method: two-stage pipeline — script router (4-class CNN) + per-script character classifier (CNN)
- Datasets: UCI Devanagari (46 classes), uTHCD Tamil (156 classes), BanglaLekha Bengali (84 classes), Telugu 6-Vowel (6 classes)
- Key results: report top-1/top-5 accuracy per script + pipeline accuracy
- Contribution: unified real-time recognition system deployed as a web app

---

### 2. Introduction (0.5 pages)

**Content:**
1. Motivation: 500M+ users of Indic scripts, digitization needs, accessibility gap
2. Challenges: high intra-class similarity (especially Tamil), multiple scripts with different complexity levels
3. Gap: most existing work treats each script independently; no unified multi-script system
4. Our contribution: two-stage pipeline, end-to-end evaluation across 4 scripts, live demo
5. Paper organization: Section 2 → Related Work, Section 3 → Data, Section 4 → Method, ...

---

### 3. Related Work (0.75 pages)

**Key papers to cite:**

| Topic | Papers |
|-------|--------|
| Devanagari recognition | Acharya et al. (UCI dataset paper), Sharma et al. (CNN-based) |
| Tamil recognition | uTHCD dataset paper (Faizal Haja Mohideen) |
| Bengali recognition | BanglaLekha dataset paper (Alam et al.) |
| Telugu recognition | Syam Kakarla (Telugu 6-Vowel dataset) |
| Multi-script recognition | Survey papers on Indic OCR |
| General CNN for character recognition | LeCun et al. (LeNet), He et al. (ResNet), Simonyan (VGG) |
| Script identification | Hangarge & Dhandra, Ferrer et al. |

**Structure:**
- Script-specific recognition (brief per script, 1–2 sentences each)
- Multi-script systems (what exists, what's missing)
- CNN architectures for fine-grained classification
- How this work differs: simultaneous 4-script, lightweight CNN, real-time app

---

### 4. Data (0.75 pages)

**Content:**
1. Table: dataset name, classes, total images, train/val/test split counts
2. Preprocessing description: resize 64×64, invert, normalize
3. Augmentation policy table
4. Class imbalance discussion (Tamil 156 classes vs Telugu 6 classes) + mitigation
5. Figure: sample grid — 4 rows (one per script) × 5 sample characters

---

### 5. Method (1.5 pages)

**Sub-sections:**

**5.1 Two-Stage Architecture Overview**
- Pipeline diagram (see architecture.md)
- Justification for two-stage vs single model (see ablation Study 5)

**5.2 Script Router (Model 1)**
- 3-block CNN architecture table
- Training details

**5.3 Character Classifier (Model 2)**
- 4-block ScriptCNN template
- Per-script instantiation (num_classes changes)

**5.4 Training Details**
- Optimizer: Adam, lr=3e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR (T_max=30)
- Early stopping: patience=5
- Hardware: Colab T4

**5.5 Inference**
- Preprocessing pipeline for canvas input
- End-to-end inference pseudocode

---

### 6. Results (1.5 pages)

**Sub-sections:**

**6.1 Script Router Performance**
- Table: Accuracy, Macro F1, per-class F1
- 4×4 Confusion matrix figure

**6.2 Per-Script Character Recognition**
- Table: Script | Top-1 Acc | Top-5 Acc | Macro F1 | Weighted F1
- One confusion matrix figure (Tamil — most complex, most interesting)
- Top confused pairs table per script

**6.3 End-to-End Pipeline**
- End-to-end accuracy table
- Error breakdown pie chart: router errors vs classifier errors

**6.4 Comparison with Baselines**
- Baseline 1: SVM on HOG features
- Baseline 2: k-NN on raw pixels
- Baseline 3: Single unified 296-class CNN (from ablation Study 5)
- Table: method × script × accuracy

**6.5 Robustness Evaluation**
- Table: Accuracy at each distortion level (noise, rotation, stroke width)

---

### 7. Ablation Study (0.75 pages)

- Reference ablation.md for full details
- Summary table: study | best variant | Δ accuracy
- Brief analysis paragraph: which factor mattered most?

**Key findings to highlight (expected):**
- Resolution 64×64 > 32×32 (meaningful gain, especially Tamil)
- BatchNorm critical for convergence speed
- Full augmentation gives +2–4% on Tamil
- Two-stage > unified 296-class model

---

### 8. Limitations (0.25 pages)

1. **Canvas vs real handwriting gap**: model trained on clean dataset images; real stylus input varies
2. **Script confusion at boundaries**: characters that look similar across scripts (e.g., Devanagari ण vs Tamil characters)
3. **Tamil class count (156)**: lower accuracy than simpler scripts; needs more data
4. **No online recognition**: uses static images, not pen-stroke sequences (DTW/RNN approaches)
5. **Deployment latency**: CPU inference is acceptable but not real-time on very slow devices

---

### 9. Conclusion (0.25 pages)

- Summary of approach and results
- Key takeaway: lightweight two-stage CNN achieves strong performance across 4 Indic scripts
- Future work: add more scripts (Telugu, Malayalam, Gujarati), online recognition, distillation for mobile

---

### 10. References (0.5 pages)

**Must-cite:**
1. UCI Devanagari dataset paper
2. uTHCD Tamil dataset paper
3. BanglaLekha-Isolated dataset paper
4. Telugu 6-Vowel Dataset (Syam Kakarla, Kaggle)
5. LeCun et al. — Gradient-based learning (LeNet, 1998)
6. He et al. — Deep Residual Learning (ResNet, 2016)
7. Ioffe & Szegedy — Batch Normalization (2015)
8. Srivastava et al. — Dropout (2014)
9. A recent Indic OCR survey paper
10. Streamlit / HuggingFace Spaces reference (for deployment)

---

## Figures Checklist

| Figure | Location in paper | Generated in |
|--------|------------------|-------------|
| Pipeline architecture diagram | Section 5.1 | Draw.io / TikZ |
| Dataset sample grid (4×5) | Section 4 | notebooks/01_data_exploration.ipynb |
| Script router confusion matrix | Section 6.1 | notebooks/04_evaluation.ipynb |
| Tamil confusion matrix (top-20 classes) | Section 6.2 | notebooks/04_evaluation.ipynb |
| End-to-end error breakdown pie chart | Section 6.3 | notebooks/04_evaluation.ipynb |
| Robustness accuracy curve | Section 6.5 | notebooks/04_evaluation.ipynb |
| Ablation summary bar chart | Section 7 | notebooks/05_ablations.ipynb |

---

## Tables Checklist

| Table | Location |
|-------|---------|
| Dataset statistics | Section 4 |
| Augmentation policy | Section 4 |
| ScriptRouter architecture | Section 5.2 |
| ScriptCNN architecture | Section 5.3 |
| Script router metrics | Section 6.1 |
| Per-script character recognition metrics | Section 6.2 |
| Comparison with baselines | Section 6.4 |
| Robustness metrics | Section 6.5 |
| Ablation summary | Section 7 |
