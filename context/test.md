# Testing & Evaluation Plan

## Evaluation Philosophy

- **Test set is held out** at all times. Never used for model selection or hyperparameter tuning.
- All reported numbers use the **test set** (15% per-script stratified split).
- The **validation set** is used only for early stopping and checkpoint selection.
- Inference runs on **CPU** for latency benchmarks (deployment target).

---

## Metrics

### Model 1: Script Router

| Metric | Computation |
|--------|-------------|
| Top-1 Accuracy | % correctly predicted script over 4 classes |
| Macro F1 | Unweighted mean F1 across 4 script classes |
| Per-class Precision / Recall / F1 | For each of 4 scripts |
| Confusion Matrix | 4×4 heatmap |

**Expected target:** ≥ 98% top-1 accuracy (scripts are visually very distinct)

### Model 2: Character Classifiers (per-script)

| Metric | Computation |
|--------|-------------|
| Top-1 Accuracy | % correctly predicted character |
| Top-5 Accuracy | % where correct class is in top-5 predictions |
| Macro F1 | Unweighted mean F1 across all character classes |
| Weighted F1 | F1 weighted by class support |
| Per-class Precision / Recall | Full per-class breakdown |
| Confusion Matrix | NxN heatmap (per script) |

**Expected targets per script:**

| Script | Top-1 Acc. (target) | Top-5 Acc. (target) |
|--------|---------------------|---------------------|
| Devanagari | ≥ 97% | ≥ 99.5% |
| Tamil | ≥ 90% | ≥ 98% |
| Bengali | ≥ 94% | ≥ 99% |
| Telugu | ≥ 98% | ≥ 100% |

Tamil is harder due to 156 classes and visual similarity of characters.

### End-to-End Pipeline

| Metric | Computation |
|--------|-------------|
| End-to-end Top-1 Accuracy | Script correct AND character correct |
| Pipeline accuracy per script | Conditioned on correct script prediction |
| Error breakdown | % errors from router vs from character classifier |

---

## Test Strategy

### Unit Tests (automated)

1. **Preprocessing tests** (`tests/test_transforms.py`):
   - Input image → output is float32 tensor of shape [1, 64, 64]
   - Normalized values in [-1, 1]
   - Inversion logic: dark background image → inverted correctly
   - Bounding box crop: image with border zeros → cropped correctly

2. **Model forward-pass tests** (`tests/test_models.py`):
   - ScriptRouter: random input [B, 1, 64, 64] → output [B, 4], all rows sum to 1
   - ScriptCNN: random input [B, 1, 64, 64] → output [B, num_classes]
   - TwoStagePipeline: single image → returns (script_id, top5_chars, top5_probs)

3. **Unicode map tests** (`tests/test_unicode.py`):
   - All class indices map to valid Unicode characters
   - No duplicate entries
   - All characters render correctly in UTF-8

### Integration Tests

4. **Full inference test**: load checkpoints → run on 100 test images per script → check output format is correct
5. **Streamlit smoke test**: app starts without errors; canvas component loads

---

## Robustness Evaluation

Run on synthetically distorted versions of the test set and report accuracy drop:

| Perturbation | Parameter | Purpose |
|--------------|-----------|---------|
| GaussianNoise | σ = 0.05, 0.10, 0.20 | Noisy scans |
| RandomRotation | ±5°, ±15°, ±30° | Tilted writing |
| StrokeWidth | Dilate 1px, 3px | Thick vs thin strokes |
| Brightness | ±20%, ±40% | Scan quality variation |
| Crop jitter | 5%, 10% border removed | Off-center characters |

Report accuracy at each distortion level in a table per script.

---

## Latency Benchmark

**Environment:** CPU inference (MacBook or Colab CPU)  
**Batch size:** 1 (single-image inference, as in the app)

| Stage | Measurement |
|-------|-------------|
| Preprocessing (raw → tensor) | ms |
| Router inference | ms |
| Character classifier inference | ms |
| Total end-to-end | ms |

Target: **< 100ms total** for a smooth real-time app experience.

Measure with `timeit.repeat(number=100)` and report mean ± std.

---

## Confusion Matrix Analysis

For each script, identify the **top-10 most confused class pairs** from the confusion matrix:

1. Extract off-diagonal entries with highest values
2. Display the actual character images for each confused pair
3. Provide a brief linguistic explanation (e.g., "क vs क़ differ only by a nukta dot")

This goes into the **Limitations** section of the report.

---

## Evaluation Notebook

All evaluation code lives in `notebooks/04_evaluation.ipynb`:

1. Load test sets per script
2. Load checkpoints
3. Run inference → collect predictions
4. Compute all metrics (sklearn.metrics)
5. Plot confusion matrices (seaborn heatmap)
6. Run robustness evaluation
7. Benchmark latency
8. Save results to `results/` as JSON + PNG figures

---

## Results Directory Structure

```
A3/results/
├── router/
│   ├── confusion_matrix.png
│   ├── classification_report.json
│   └── metrics.json
├── devanagari/
│   ├── confusion_matrix.png
│   ├── classification_report.json
│   ├── metrics.json
│   └── robustness.json
├── tamil/
├── bengali/
├── telugu/
└── pipeline/
    ├── end_to_end_metrics.json
    └── error_breakdown.json
```
