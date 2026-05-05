# Timeline & Milestones

## Overview

Total estimated time: **~14 days** (working ~3–4 hours/day)

---

## Phase Breakdown

### Phase 1 — Data Preparation (Days 1–2)

| Task | Time | Output |
|------|------|--------|
| Download all 4 datasets | 1–2 hrs | `data/raw/` |
| Write preprocessing script (resize, invert, crop, save) | 2 hrs | `src/data/datasets.py`, `src/data/transforms.py` |
| Generate processed dataset in `data/processed/` | 1 hr | 64×64 PNG files per script |
| Build router dataset (5k per script, balanced) | 1 hr | `data/processed/router/` |
| Data exploration notebook (stats, sample grid, histograms) | 1 hr | `notebooks/01_data_exploration.ipynb` |
| Build Unicode maps for all 4 scripts | 1 hr | `src/utils/unicode_maps.py` |

**Milestone 1:** All data is preprocessed, split, and verified. Unicode maps complete.

---

### Phase 2 — Model Implementation (Days 3–4)

| Task | Time | Output |
|------|------|--------|
| Implement ScriptRouter CNN | 1 hr | `src/models/script_router.py` |
| Implement ScriptCNN (character classifier) | 1 hr | `src/models/char_classifier.py` |
| Implement TwoStagePipeline wrapper | 1 hr | `src/models/pipeline.py` |
| Implement preprocessing + augmentation transforms | 1 hr | `src/data/transforms.py` |
| Implement metrics helpers (accuracy, F1, confusion matrix) | 1 hr | `src/utils/metrics.py` |
| Write unit tests (model forward pass, transforms) | 1 hr | `tests/` |

**Milestone 2:** All model code implemented and unit-tested. Forward passes verified.

---

### Phase 3 — Training (Days 5–7)

| Task | Time | Output |
|------|------|--------|
| Train Script Router | 20 min (Colab T4) | `checkpoints/script_router.pth` |
| Train Devanagari classifier (46 classes) | 30 min | `checkpoints/devanagari_classifier.pth` |
| Train Bengali classifier (84 classes) | 40 min | `checkpoints/bengali_classifier.pth` |
| Train Telugu classifier (6 classes) | 15 min | `checkpoints/telugu_classifier.pth` |
| Train Tamil classifier (156 classes) | 60 min | `checkpoints/tamil_classifier.pth` |
| Monitor training curves, fix any issues | 2 hrs | Training logs |
| Save final checkpoints + training notebooks | 30 min | `notebooks/02_train_router.ipynb`, `notebooks/03_train_classifiers.ipynb` |

**Milestone 3:** All 5 models trained and checkpoints saved. Val accuracy meets targets.

**Target checkpoints:**
- Script Router: ≥ 98% val accuracy
- Devanagari: ≥ 97%
- Tamil: ≥ 90%
- Bengali: ≥ 94%
- Telugu: ≥ 98%

---

### Phase 4 — Evaluation (Days 8–9)

| Task | Time | Output |
|------|------|--------|
| Run test-set evaluation for all models | 1 hr | `results/*/metrics.json` |
| Generate confusion matrices | 1 hr | `results/*/confusion_matrix.png` |
| Compute end-to-end pipeline accuracy + error breakdown | 1 hr | `results/pipeline/` |
| Run robustness evaluation (5 distortions × 4 scripts) | 2 hrs | `results/*/robustness.json` |
| CPU latency benchmark | 30 min | Latency numbers |
| Identify top-10 confused pairs per script | 1 hr | Analysis notes |
| Finalize `notebooks/04_evaluation.ipynb` | 30 min | Complete eval notebook |

**Milestone 4:** Full evaluation complete. All metrics available for report.

---

### Phase 5 — Ablation Studies (Days 9–11)

| Task | Time | Output |
|------|------|--------|
| Run Study 1: CNN depth (4 variants) | 40 min | Ablation logs |
| Run Study 2: Input resolution (4 variants) | 40 min | Ablation logs |
| Run Study 3: BatchNorm (2 variants) | 20 min | Ablation logs |
| Run Study 4: Dropout rate (4 variants) | 40 min | Ablation logs |
| Run Study 5: Pipeline design (3 variants) | 60 min | Ablation logs |
| Run Study 6: Augmentation (4 variants) | 40 min | Ablation logs |
| Run Study 7: LR scheduler (4 variants) | 40 min | Ablation logs |
| Compile ablation summary table + bar chart | 1 hr | `results/ablations/` |
| Finalize `notebooks/05_ablations.ipynb` | 30 min | Complete ablation notebook |

**Milestone 5:** All 7 ablation studies complete. Summary table filled in.

---

### Phase 6 — Streamlit App (Days 11–12)

| Task | Time | Output |
|------|------|--------|
| Implement `app/inference.py` (preprocessing + model load + predict) | 2 hrs | `app/inference.py` |
| Implement main `app/app.py` (canvas + prediction UI) | 3 hrs | `app/app.py` |
| Test app locally with all 4 scripts | 1 hr | Working local demo |
| Implement practice mode | 2 hrs | Practice mode UI |
| Polish UI (colors, layout, confidence bars) | 1 hr | Final app |

**Milestone 6:** App runs locally, all 4 scripts work, practice mode functional.

---

### Phase 7 — Deployment (Day 12)

| Task | Time | Output |
|------|------|--------|
| Create public GitHub repo, push all code | 1 hr | GitHub repo |
| Write `README.md` | 1 hr | README |
| Create HuggingFace Space, push app | 1 hr | Live demo URL |
| Test live deployment on HuggingFace | 30 min | Verified live demo |
| Record 2-min screen demo (backup) | 30 min | Demo video |

---

## Session Log (Completed work)

### Bug Fixes (smoke test parity)
- **transforms.py**: `RandomErasing` moved after `ToTensor()` — requires tensor input
- **datasets.py**: `make_loader()` guard — `drop_last` disabled when `len(dataset) < batch_size`
- **trainer.py / evaluate.py / models.py**: MPS crash fix — `non_blocking=True` only on CUDA;
  `@torch.no_grad()` decorator crashes MPS, replaced with `with torch.no_grad():` context manager
- **smoke_test.py**: all 8 checks pass (exit 0)

### Architecture Changes
- **src/config.py**: Added `KAGGLE_SMOKE` mode (set via `os.environ["KAGGLE_SMOKE"]="1"`
  before importing config). Triggers `_SMALL_RUN=True`:
  - `ROUTER_EPOCHS=3`, `CHAR_EPOCHS=3`, `PATIENCE=2`
  - `CHAR_MAX=50` (max imgs/class), `ROUTER_MAX=200`
  - `RUN_ABLATIONS=False`
  - `TEST_N_PER_CLASS` removed, replaced by `CHAR_MAX`

### Kaggle Setup (✓ DONE)
- **train.ipynb**: Rewritten as thin 21-cell notebook delegating entirely to `src/` modules
  - Cell 2 has `KAGGLE_SMOKE_RUN = True` switch (set BEFORE importing config)
  - Completes in ~15 min on 2×T4 in smoke mode
- **kernel-metadata.json**: Created with 3 Kaggle dataset slugs + GPU enabled
- **Pushed**: `kaggle kernels push -p .` → version 1 running
  - URL: https://www.kaggle.com/code/dhruv10050/t3-7-multi-script-indic-handwriting-recognition
  - Status: RUNNING (check with `kaggle kernels status dhruv10050/t3-7-multi-script-indic-handwriting-recognition`)

**Milestone 7:** Live demo accessible. GitHub repo public and complete.

---

### Phase 8 — Report (Days 12–14)

| Task | Time | Output |
|------|------|--------|
| Write Introduction + Related Work | 2 hrs | Sections 1–3 |
| Write Data section + insert figures | 1 hr | Section 4 |
| Write Method section + architecture diagram | 2 hrs | Section 5 |
| Write Results section + tables + figures | 2 hrs | Section 6 |
| Write Ablation section | 1 hr | Section 7 |
| Write Limitations + Conclusion | 1 hr | Sections 8–9 |
| Compile references | 30 min | Section 10 |
| LaTeX formatting, proofread, export PDF | 2 hrs | Final PDF |
| Create one-slide pitch (Canva/Figma) | 1 hr | `pitch.png` |

**Milestone 8:** Report PDF complete (6–8 pages). Pitch slide ready.

---

## Actual Status (2026-05-05)

All planning phases are complete. The project moved to a single-notebook Kaggle execution model rather than the original multi-script phased plan.

| Phase | Status | Notes |
|-------|--------|-------|
| Data download | ✅ Done | All 4 datasets downloaded locally and available as Kaggle dataset sources |
| Model implementation | ✅ Done | `src/models.py` — ScriptRouter + ScriptCNN |
| Training loop | ✅ Done | `src/trainer.py` — AMP probe, CSV logging, checkpointing, curve plotting |
| Datasets pipeline | ✅ Done | `src/datasets.py` — all 4 scripts + router loader |
| Transforms | ✅ Done | `src/transforms.py` |
| Evaluation | ✅ Done | `src/evaluate.py` — per-script, robustness, latency |
| Ablations | ✅ Done | `src/ablations.py` — 7 studies on Devanagari |
| Config | ✅ Done | `src/config.py` — ENV detection, smoke/full hyperparams |
| Notebook | ✅ Done | `train.ipynb` — 21 cells, local smoke passes |
| README | ✅ Done | `README.md` created |
| Kaggle push infra | ✅ Done | `_push_dataset2.py`, `_push_kernel.py`, `_monitor.py` |
| Kaggle smoke run | ✅ Done | v6 completed successfully |
| Kaggle full run | 🔄 In progress | v12 running — CUDA compatibility fix applied |
| Report | ⬜ Pending | Awaiting full-run results |

---

## Summary Timeline

| Day | Phase | Key Deliverable |
|-----|-------|----------------|
| 1 | Data | Datasets downloaded |
| 2 | Data | Processed data + Unicode maps |
| 3 | Models | Model code implemented |
| 4 | Models | Unit tests passing |
| 5 | Training | Router + Devanagari trained |
| 6 | Training | Bengali + Telugu trained |
| 7 | Training | Tamil trained, all checkpoints saved |
| 8 | Evaluation | Test metrics + confusion matrices |
| 9 | Evaluation + Ablation | Robustness eval + first 3 ablation studies |
| 10 | Ablation | Remaining ablation studies |
| 11 | Kaggle | Full run + output collection |
| 12 | Report | First draft |
| 13 | Report | Final polished PDF |
| 14 | Submission | Push + submit |

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Tamil model < 90% accuracy | Medium | Add weighted sampling, stronger augmentation, increase epochs |
| Kaggle GPU SM incompatibility | ✅ Mitigated | CUDA probe in cell 2; install torch<2.7+cu121 if P100 SM 6.0 detected |
| uTHCD dataset download issues | Low | Available as Kaggle dataset source |
| Full run time > 12hr Kaggle limit | Low | Smoke run was ~15 min; full run estimated 5–6 hrs |
| Colab T4 session timeout during training | Low | Save checkpoints every 5 epochs; resume from checkpoint |
