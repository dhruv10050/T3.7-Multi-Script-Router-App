# V1 System Report - T3.7

Date: 2026-05-05  
Scope reviewed: context docs, src modules, notebook orchestration, and repo deliverables.

This report lists issues found in the current codebase and approach, grouped by severity.

---

## Patch Status (updated 2026-05-05)

| Issue | Status | Fix applied |
|-------|--------|-------------|
| C1 – Credential leak | ⚠️ Open | `temp.txt` still tracked; tokens need rotation and `git filter-repo` purge |
| C2 – Test-set leakage | ⬜ Open | Not yet addressed |
| C3 – Kannada vs Telugu scope | ✅ Fixed | All planning docs updated to Telugu; implementation was already Telugu |
| C4 – Pipeline ablation naming | ⬜ Open | Study 7 ("pipeline design") is still architecture depth only |
| M1 – Smoke default encourages wrong metrics | ✅ Fixed | Default is `KAGGLE_SMOKE_RUN = False`; smoke requires explicit flag |
| M2 – Non-deterministic runtime installs | ⬜ Partial | `requirements.txt` cleaned (torch/torchvision removed); install cell still uses silent pip |
| M3 – Weighted sampler expensive | ⬜ Open | `weighted_sampler` in `datasets.py` still iterates items |
| M4 – Stochastic robustness eval | ⬜ Open | PERTURBATIONS dict uses fixed params but RandomAffine still draws at eval time |
| M5 – Inconsistent E2E latency | ⬜ Open | `run_latency_benchmark` accumulates last script latency only |
| M6 – Router no independent test eval | ⬜ Open | Router evaluation still uses same val split |
| L1 – Missing jpeg support | ⬜ Open | `_collect_img_samples` still png/bmp only |
| L2 – EDA png-only glob | ⬜ Open | EDA grid still uses png glob |
| L3 – blur_light no-op | ⬜ Open | GaussianBlur(1) still present |
| L4 – Smoke flag name mismatch | ✅ Fixed | `smoke_test.py` removed; `KAGGLE_SMOKE` is the sole canonical flag |
| L5 – Docs Kannada vs Telugu | ✅ Fixed | All context/*.md updated |

### Additional issues discovered post-v1

| ID | Severity | Description | Status |
|----|----------|-------------|--------|
| A1 | Critical | CUDA kernel incompatibility on Kaggle P100 (SM 6.0): torch ≥2.7+cu128 drops SM 6.0 support | ✅ Fixed — cell 2 probes CUDA via subprocess and installs `torch<2.7+cu121` if needed; `_amp_supported()` in `trainer.py` also probes fp16 and falls back to fp32 |
| A2 | High | `docker_image_pinning_type: "latest"` made things worse by pulling torch 2.10+cu128 | ✅ Fixed — reverted to `"original"` |
| A3 | High | In-process `sys.modules` purge for torch fails because C extensions cannot be reloaded | ✅ Fixed — probe runs in subprocess; new wheel is picked up by later cells without purge |

---

## Critical Issues

### C1. Plain-text credential leak committed in repository
Evidence:
- [temp.txt](temp.txt#L2)
- [temp.txt](temp.txt#L5)
- [temp.txt](temp.txt#L8)

Issue:
- Active-looking Kaggle and Hugging Face tokens are stored in a tracked text file.

Consequences:
- Immediate risk of account compromise, unauthorized dataset/model actions, billing abuse, and irreversible data tampering.
- Even if deleted now, leaked secrets may remain in Git history and caches.

Fix / Solution:
- Revoke and rotate all exposed tokens immediately.
- Remove the file from the repo and rewrite history to purge secrets from previous commits.
- Add secret scanning in CI (for example, gitleaks or trufflehog) and pre-commit hooks.
- Add explicit ignore patterns for local scratch files that may hold secrets.

---

### C2. Test-set leakage into model selection (Devanagari and Tamil)
Evidence:
- Devanagari validation and test both read from DEVA_TEST:
  - [../train.ipynb](../train.ipynb#L517)
  - [../train.ipynb](../train.ipynb#L520)
- Tamil validation and test both read from TAMIL_TEST:
  - [../train.ipynb](../train.ipynb#L532)
  - [../train.ipynb](../train.ipynb#L535)
- Ablation also uses DEVA_TEST for tuning:
  - [../train.ipynb](../train.ipynb#L799)

Issue:
- The same official test split is used for validation (early stopping/model selection) and then again for final reporting.

Consequences:
- Reported test metrics are optimistically biased.
- Invalidates strict holdout protocol expected in Tier-2 evaluation.
- Makes comparisons with baselines and ablations scientifically weak.

Fix / Solution:
- Build validation only from train split (for example, train->train/val stratified split).
- Keep official test split untouched until final one-time evaluation.
- Run ablations only on validation, then evaluate best configuration once on test.

---

### C3. Core assignment scope mismatch (required Kannada, implemented Telugu)
Evidence:
- Assignment requirement includes Kannada-MNIST:
  - [assignment.md](assignment.md#L14)
- Implementation is Telugu pipeline and datasets:
  - [../train.ipynb](../train.ipynb#L10)
  - [../train.ipynb](../train.ipynb#L290)
  - [../train.ipynb](../train.ipynb#L553)

Issue:
- The implemented task deviates from the stated assignment dataset scope.

Consequences:
- Risk of rubric non-compliance and grading penalty.
- Report claims may be judged against the wrong benchmark/task definition.

Fix / Solution:
- Either migrate back to Kannada-MNIST end-to-end, or
- document and obtain explicit approval for the Telugu substitution,
- and update all context/report artifacts to one consistent scope.

---

### C4. Claimed pipeline-design ablation is not actually pipeline design
Evidence:
- Study label says pipeline design, but variants are only shallow vs deep ScriptCNN:
  - [../src/ablations.py](../src/ablations.py#L220)
  - [../src/ablations.py](../src/ablations.py#L221)

Issue:
- The code does not compare true two-stage pipeline vs true unified model, despite naming it that way.

Consequences:
- Misleading ablation conclusions.
- Weakens credibility of the final report and any "two-stage is better" claim.

Fix / Solution:
- Implement actual unified single-model baseline across all classes.
- Compare against true two-stage under matched data and training budget.
- Rename current study if it is only architecture depth, not pipeline design.

## Medium Issues

### M1. Default execution path encourages smoke-mode results
Evidence:
- Notebook default sets smoke run ON:
  - [../train.ipynb](../train.ipynb#L72)
- Config forces small-run on all mac/cpu executions:
  - [../src/config.py](../src/config.py#L35)
  - [../src/config.py](../src/config.py#L37)

Issue:
- Easy to accidentally produce/report smoke metrics as if they are final.

Consequences:
- Undertrained checkpoints, lower quality figures, and misleading performance conclusions.

Fix / Solution:
- Default notebook to full mode and require explicit smoke override.
- Decouple TEST_RUN from hardware type; make it an explicit environment flag.

---

### M2. Runtime dependency installation is non-deterministic and silent on failure
Evidence:
- Runtime pip installs in notebook startup:
  - [../train.ipynb](../train.ipynb#L45)

Issue:
- Dependencies are installed ad hoc during execution; return codes are ignored.

Consequences:
- Reproducibility drift across runs.
- Hidden setup failures can surface later as confusing runtime errors.

Fix / Solution:
- Move dependencies to pinned requirements file and lock environment.
- In notebook, either remove installs or enforce check=True and fail fast.

---

### M3. Weighted sampler construction is unnecessarily expensive
Evidence:
- Sampler computes labels by calling dataset item access for every sample:
  - [../src/datasets.py](../src/datasets.py#L297)

Issue:
- Building the sampler can trigger image loads/transforms for entire datasets.

Consequences:
- Large startup overhead for Tamil/Bengali scale.
- Avoidable IO and preprocessing cost every run.

Fix / Solution:
- Read labels from dataset metadata arrays directly (for example, dataset.labels or dataset.samples) without image decoding.
- Cache class counts/sampler weights once per dataset instance.

---

### M4. Robustness evaluation is stochastic and not reproducible
Evidence:
- Random perturbation transforms are used during evaluation:
  - [../src/evaluate.py](../src/evaluate.py#L143)
  - [../src/evaluate.py](../src/evaluate.py#L144)
  - [../src/evaluate.py](../src/evaluate.py#L146)

Issue:
- Metrics depend on random transform draws and can vary run to run.

Consequences:
- Unstable robustness tables, difficult comparison across experiments.

Fix / Solution:
- Replace random transforms with deterministic variants (fixed angle, fixed crop offsets), or lock transform RNG and log seeds.

---

### M5. End-to-end latency estimate is logically inconsistent
Evidence:
- Uses router latency + latency of the last iterated script model only:
  - [../src/evaluate.py](../src/evaluate.py#L238)
  - [../src/evaluate.py](../src/evaluate.py#L250)

Issue:
- End-to-end figure depends on loop order rather than a clear policy (mean, weighted mean, or worst-case).

Consequences:
- Latency claim can be biased and non-representative.

Fix / Solution:
- Report per-script E2E latency and aggregate with explicit policy.
- Use weighted average by script frequency and also report worst-case.

---

### M6. Router has no independent test evaluation
Evidence:
- Router reports validation classification report only:
  - [../train.ipynb](../train.ipynb#L452)

Issue:
- Router performance is selected and reported on the same split family.

Consequences:
- End-to-end error attribution (router vs classifier) is less trustworthy.

Fix / Solution:
- Create dedicated router test split untouched during router training.
- Report router test metrics and confusion matrix separately.

---

### M7. Import strategy is brittle outside notebook path hacks
Evidence:
- src modules use top-level imports rather than package-relative imports:
  - [../src/evaluate.py](../src/evaluate.py#L29)
  - [../src/datasets.py](../src/datasets.py#L29)
  - [../src/trainer.py](../src/trainer.py#L27)

Issue:
- Code works only when src is manually injected into sys.path, reducing portability.

Consequences:
- Fragile execution in tests, scripts, packaging, and deployment.

Fix / Solution:
- Convert to package-relative imports (for example, from .config import ...).
- Use proper package/module entrypoints instead of sys.path mutation.

---

### M8. Required repository deliverables are incomplete
Evidence:
- Assignment requires README, requirements, and app code:
  - [assignment.md](assignment.md#L21)
- In current tree, README, requirements, app, and tests folders are absent.

Issue:
- Submission structure does not satisfy stated deliverables.

Consequences:
- Reduced reproducibility and likely grading penalties despite working notebook.

Fix / Solution:
- Add README with setup/train/eval/deploy sections.
- Add pinned requirements file.
- Add minimal Streamlit app code and tests directory with smoke/inference tests.

## Low Issues

### L1. Router image collector misses JPEG extension variant
Evidence:
- Collector includes png/jpg/bmp but not jpeg:
  - [../src/datasets.py](../src/datasets.py#L310)

Issue:
- Some datasets with .jpeg files may be partially ignored.

Consequences:
- Silent data loss and inconsistent router training size.

Fix / Solution:
- Include .jpeg in _collect_img_samples.

---

### L2. EDA grid reader only pulls png from folder datasets
Evidence:
- EDA folder path uses png glob only:
  - [../train.ipynb](../train.ipynb#L348)

Issue:
- Tamil bmp data may not be visualized in EDA, giving blank or misleading sample grids.

Consequences:
- Incorrect qualitative inspection and potential false confidence in data integrity.

Fix / Solution:
- Extend EDA image search to png/jpg/jpeg/bmp.

---

### L3. One robustness setting is effectively a no-op
Evidence:
- blur_light uses GaussianBlur(1):
  - [../src/evaluate.py](../src/evaluate.py#L141)

Issue:
- Kernel size 1 does not meaningfully blur the image.

Consequences:
- Reported robustness on this perturbation is not informative.

Fix / Solution:
- Use meaningful blur kernels (for example, 3 and 5) with fixed sigma.

---

### L4. Smoke-test flag mismatch
Evidence:
- Smoke test sets SMOKE_TEST env var:
  - [../smoke_test.py](../smoke_test.py#L22)
- Config reads KAGGLE_SMOKE, not SMOKE_TEST:
  - [../src/config.py](../src/config.py#L36)

Issue:
- Intended smoke control variable is unused.

Consequences:
- Confusing behavior and hidden coupling to hardware-based TEST_RUN logic.

Fix / Solution:
- Use one canonical flag name across smoke_test and config.

---

### L5. Documentation set is internally inconsistent (Telugu vs Kannada)
Evidence:
- Kannada still appears in planning/evaluation docs:
  - [report.md](report.md#L24)
  - [test.md](test.md#L43)
  - [deployment.md](deployment.md#L32)
  - [timeline.md](timeline.md#L48)

Issue:
- Context docs and implementation are out of sync.

Consequences:
- Reviewer confusion, inconsistent report narrative, and weak reproducibility.

Fix / Solution:
- Align all context docs to one final task definition and dataset set.

## Recommended Immediate Order of Execution

1. Revoke and rotate leaked keys, remove secrets from history.
2. Fix data leakage (strict train/val/test protocol) and rerun all training/evaluation.
3. Resolve dataset-scope mismatch (Kannada vs Telugu) and align docs.
4. Rework pipeline-design ablation to a true baseline comparison.
5. Harden reproducibility: pinned dependencies, deterministic robustness, proper imports.