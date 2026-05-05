# Ablation Study Plan

## Purpose

Ablation studies isolate the contribution of each design decision. Each study changes **one variable** while keeping all others at the default configuration. Results are reported on the **validation set** of Devanagari (46 classes, representative complexity) to keep compute manageable, with the final winner verified on all 4 scripts.

**Default (baseline) configuration:**
- Input resolution: 64×64
- CNN depth: 4 blocks (for character classifier), 3 blocks (for router)
- Filters: 32→64→128→256
- BatchNorm: enabled
- Dropout rate: 0.5
- Augmentation: enabled (rotation, affine, erasing, noise)
- LR scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss (unweighted)
- Optimizer: Adam, lr=3e-4

---

## Study 1: CNN Depth (Number of Conv Blocks)

**Hypothesis:** More layers capture finer stroke details but risk overfitting on small scripts.

| Variant | Conv Blocks | Filter Progression | Approx. Params |
|---------|------------|-------------------|----------------|
| 2-layer | 2 | 32→64 | ~0.5M |
| 3-layer | 3 | 32→64→128 | ~1.2M |
| **4-layer (default)** | **4** | **32→64→128→256** | **~2.8M** |
| 5-layer | 5 | 32→64→128→256→256 | ~4.2M |

**Result table:**

| Variant | Val Acc. | Val F1 (macro) | Params |
|---------|----------|----------------|--------|
| 2-layer | | | |
| 3-layer | | | |
| 4-layer (default) | | | |
| 5-layer | | | |

---

## Study 2: Input Resolution

**Hypothesis:** Higher resolution preserves fine strokes but increases compute; 64×64 may be sufficient.

| Variant | Resolution | Notes |
|---------|-----------|-------|
| 28×28 | 28×28 | Native Devanagari dataset size |
| 32×32 | 32×32 | Native Devanagari size |
| **64×64 (default)** | **64×64** | Upsampled for all datasets |
| 96×96 | 96×96 | Higher detail |

**All other hyperparameters held constant. FC input size changes accordingly.**

| Variant | Val Acc. | Inference (ms) |
|---------|----------|---------------|
| 28×28 | | |
| 32×32 | | |
| 64×64 (default) | | |
| 96×96 | | |

---

## Study 3: Batch Normalization

**Hypothesis:** BatchNorm stabilizes training and acts as a regularizer; removing it degrades performance.

| Variant | BatchNorm | Notes |
|---------|-----------|-------|
| No BN | ✗ | Baseline without BN |
| **With BN (default)** | **✓** | After each Conv layer |
| BN + GroupNorm | ✓ + GN | BN in early layers, GN in later |

| Variant | Val Acc. | Val Loss | Epochs to converge |
|---------|----------|----------|--------------------|
| No BN | | | |
| With BN (default) | | | |

---

## Study 4: Dropout Rate

**Hypothesis:** Higher dropout is needed for Tamil (156 classes, fewer samples/class) but hurts Telugu (6 classes, many samples).

| Variant | Dropout p |
|---------|-----------|
| No dropout | 0.0 |
| Low | 0.3 |
| **Default** | **0.5** |
| High | 0.7 |

| Dropout | Val Acc. | Val F1 | Train-Val Acc. Gap |
|---------|----------|--------|-------------------|
| 0.0 | | | |
| 0.3 | | | |
| 0.5 (default) | | | |
| 0.7 | | | |

---

## Study 5: Pipeline Design — Two-stage vs Single Unified Model

**Hypothesis:** Two-stage (route then classify) is more accurate and modular than one monolithic model.

| Variant | Description | Total Classes |
|---------|-------------|--------------|
| **Two-stage (default)** | Router (4 classes) → per-script CNN | 4 + {46/156/84/10} |
| Single unified | One CNN for all 296 classes | 296 |
| Two-stage (shared backbone) | Shared CNN trunk, 4 script-specific heads | 4+N |

| Variant | End-to-end Acc. | Params | Training Time |
|---------|----------------|--------|---------------|
| Two-stage (default) | | | |
| Single unified (296-class) | | | |
| Shared backbone | | | |

---

## Study 6: Data Augmentation

**Hypothesis:** Augmentation is critical for Tamil (few samples/class) and less critical for Telugu (many samples/class).

Test on Devanagari and Tamil (the most different in terms of samples/class).

| Variant | Rotation | Affine | Erasing | Noise |
|---------|----------|--------|---------|-------|
| No augmentation | ✗ | ✗ | ✗ | ✗ |
| Rotation only | ✓ | ✗ | ✗ | ✗ |
| Rotation + Affine | ✓ | ✓ | ✗ | ✗ |
| **Full (default)** | **✓** | **✓** | **✓** | **✓** |

| Variant | Devanagari Val Acc. | Tamil Val Acc. |
|---------|---------------------|----------------|
| No augmentation | | |
| Rotation only | | |
| Rotation + Affine | | |
| Full (default) | | |

---

## Study 7: Learning Rate Scheduler

**Hypothesis:** CosineAnnealingLR prevents premature convergence vs StepLR; no scheduler underfits.

| Variant | Scheduler | Config |
|---------|-----------|--------|
| No scheduler | None | Fixed lr=3e-4 |
| StepLR | StepLR | step_size=10, gamma=0.5 |
| **CosineAnnealingLR (default)** | **CosineAnnealingLR** | **T_max=30** |
| ReduceLROnPlateau | ReduceLROnPlateau | factor=0.5, patience=3 |

| Variant | Val Acc. | Best Epoch | Final LR |
|---------|----------|-----------|---------|
| No scheduler | | | |
| StepLR | | | |
| CosineAnnealingLR (default) | | | |
| ReduceLROnPlateau | | | |

---

## Summary Table (to include in report)

Fill in after running all studies:

| Study | Best Variant | Δ Accuracy vs Baseline |
|-------|-------------|------------------------|
| 1. CNN Depth | | |
| 2. Input Resolution | | |
| 3. BatchNorm | | |
| 4. Dropout | | |
| 5. Pipeline Design | | |
| 6. Augmentation | | |
| 7. LR Scheduler | | |

---

## Implementation Notes

- All ablation experiments use the same random seed (42) and train for the same number of epochs
- Save each variant's best checkpoint to `checkpoints/ablations/`
- Log all runs with a simple CSV logger (columns: study, variant, epoch, train_loss, val_loss, val_acc, val_f1)
- Ablation training script: `notebooks/05_ablations.ipynb`
- Each study should take < 10 minutes on Colab T4

---

## Compute Budget Estimate

| Study | Variants | Epochs | Est. Time (Colab T4) |
|-------|----------|--------|----------------------|
| 1. Depth | 4 | 30 | ~40 min |
| 2. Resolution | 4 | 30 | ~40 min |
| 3. BatchNorm | 2 | 30 | ~20 min |
| 4. Dropout | 4 | 30 | ~40 min |
| 5. Pipeline | 3 | 30 | ~60 min |
| 6. Augmentation | 4 | 30 | ~40 min |
| 7. LR Scheduler | 4 | 30 | ~40 min |
| **Total** | **25** | — | **~5.5 hrs** |
