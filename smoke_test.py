#!/usr/bin/env python3
"""
smoke_test.py — Validates the full pipeline using 100 % synthetic data.

Creates tiny temp image folders + CSV for each script (no real data needed),
then runs: transforms → datasets → loaders → router training (1 epoch) →
char classifier training (1 epoch) → evaluation → ablation (1 epoch).

Exit 0 on success, 1 on any failure.
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path

# ── Bootstrap: add src/ to path ───────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

# Patch config before any imports so we get CPU / tiny settings
os.environ.setdefault("KAGGLE_SMOKE", "1")

import numpy as np
from PIL import Image
import pandas as pd
import torch

print("=" * 60)
print("SMOKE TEST — Indic Handwriting Recognition Pipeline")
print("=" * 60)

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        import traceback; traceback.print_exc()
        FAILURES.append(name)


# ── Create synthetic datasets ─────────────────────────────────────────────────

TMP = Path(tempfile.mkdtemp(prefix="smai_smoke_"))

def _make_img_folder(root: Path, n_classes: int, n_per: int, size: int = 32):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for c in range(n_classes):
        d = root / f"class_{c:03d}"
        d.mkdir()
        for i in range(n_per):
            arr = rng.randint(0, 256, (size, size), dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(d / f"img_{i}.png")

def _make_csv(path: Path, n_samples: int, n_classes: int, side: int = 16):
    rng = np.random.RandomState(0)
    px  = rng.randint(0, 256, (n_samples, side * side), dtype=np.uint8)
    lbls = rng.randint(0, n_classes, n_samples)
    df   = pd.DataFrame(px, columns=[f"p{i}" for i in range(side * side)])
    df["label"] = lbls
    df.to_csv(path, index=False)

N_CLS  = {"deva": 6, "tamil": 8, "bengali": 6, "telugu": 6}
N_IMG  = 12   # images per class (must be > 3 for stratified split)
IMG_SZ = 32

DEVA_ROOT   = TMP / "deva"
TAMIL_ROOT  = TMP / "tamil"
BENGALI_ROOT= TMP / "bengali"
TELUGU_CSV  = TMP / "telugu.csv"

print("\nCreating synthetic datasets …")
_make_img_folder(DEVA_ROOT   / "Train", N_CLS["deva"],    N_IMG, IMG_SZ)
_make_img_folder(DEVA_ROOT   / "Test",  N_CLS["deva"],    4,     IMG_SZ)
_make_img_folder(TAMIL_ROOT  / "train", N_CLS["tamil"],   N_IMG, IMG_SZ)
_make_img_folder(TAMIL_ROOT  / "test",  N_CLS["tamil"],   4,     IMG_SZ)
_make_img_folder(BENGALI_ROOT,          N_CLS["bengali"],
                 int(N_IMG / 0.7) + 1, IMG_SZ)  # enough for 70/15/15 split
_make_csv(TELUGU_CSV, n_samples=120, n_classes=N_CLS["telugu"])
print("  done.")

# ── 1. Config & utils ─────────────────────────────────────────────────────────

def test_config():
    from config import ENV, DEVICE, WORK_DIR, print_config
    print_config()
    assert WORK_DIR.exists()

def test_utils():
    from utils import seed_everything, free_memory, wrap_model, unwrap, find_path
    seed_everything(42)
    free_memory()
    from models import ScriptRouter
    m = ScriptRouter()
    wrapped = wrap_model(m)
    assert unwrap(wrapped) is m or hasattr(wrapped, "module")
    found = find_path(str(TMP), "/nonexistent/path")
    assert found == TMP

check("config + utils", lambda: (test_config(), test_utils()))

# ── 2. Transforms ─────────────────────────────────────────────────────────────

def test_transforms():
    from transforms import get_transforms
    from PIL import Image as _I
    import numpy as np
    for mode in ["train", "val"]:
        tfm = get_transforms(mode, 64)
        img = _I.fromarray(np.random.randint(0, 255, (32, 32), dtype=np.uint8), mode="L")
        t = tfm(img)
        assert t.shape == (1, 64, 64), f"wrong shape: {t.shape}"

check("transforms", test_transforms)

# ── 3. Dataset classes ────────────────────────────────────────────────────────

def test_datasets():
    from transforms import get_transforms
    from datasets import (ScriptImageDataset, BanglaLekhaDataset,
                          TeluguCSVDataset, ListDataset)

    ds = ScriptImageDataset(DEVA_ROOT / "Train", transform=get_transforms("train"),
                            max_per_class=5)
    assert len(ds) > 0
    img, lbl = ds[0]
    assert img.shape[0] == 1

    # Bengali split
    bd = BanglaLekhaDataset(BENGALI_ROOT, split="train",
                             transform=get_transforms("train"), max_per_class=5)
    assert len(bd) > 0
    assert bd.num_classes == N_CLS["bengali"]

    # Telugu CSV
    td = TeluguCSVDataset(TELUGU_CSV, split="train", transform=get_transforms("train"))
    assert len(td) > 0
    assert td.num_classes == N_CLS["telugu"]

    # ListDataset
    samples = [(str(p), 0) for p in list((DEVA_ROOT / "Train" / "class_000").glob("*.png"))[:3]]
    ld = ListDataset(samples, transform=get_transforms("val"))
    assert len(ld) == 3

check("dataset classes", test_datasets)

# ── 4. DataLoaders ────────────────────────────────────────────────────────────

def test_loaders():
    from transforms import get_transforms
    from datasets import (ScriptImageDataset, make_loader,
                          weighted_sampler, build_router_loaders)

    ds  = ScriptImageDataset(DEVA_ROOT / "Train", transform=get_transforms("train"),
                              max_per_class=5)
    dl  = make_loader(ds, batch_size=4, shuffle=True)
    imgs, lbls = next(iter(dl))
    assert imgs.shape[1] == 1

    tr_dl, vl_dl, cw = build_router_loaders(
        deva_train  = DEVA_ROOT / "Train",
        tamil_train = TAMIL_ROOT / "train",
        bengali_dir = BENGALI_ROOT,
        telugu_csv  = TELUGU_CSV,
        max_per_script = 30,
    )
    imgs, lbls = next(iter(tr_dl))
    assert imgs.shape[1] == 1
    assert lbls.max() <= 3

check("dataloaders", test_loaders)

# ── 5. Models ─────────────────────────────────────────────────────────────────

def test_models():
    from models import ScriptRouter, ScriptCNN, TwoStagePipeline
    from config import DEVICE
    x = torch.randn(2, 1, 64, 64).to(DEVICE)

    r = ScriptRouter(4).to(DEVICE)
    assert r(x).shape == (2, 4)

    for nl in [2, 3, 4]:
        c = ScriptCNN(10, nl).to(DEVICE)
        assert c(x).shape == (2, 10), f"ScriptCNN(nl={nl}) failed"

    pipeline = TwoStagePipeline(
        ScriptRouter(4).to(DEVICE),
        {"devanagari": ScriptCNN(6, 4).to(DEVICE),
         "tamil":      ScriptCNN(8, 4).to(DEVICE),
         "bengali":    ScriptCNN(6, 4).to(DEVICE),
         "telugu":     ScriptCNN(6, 3).to(DEVICE)},
    ).to(DEVICE)
    out = pipeline(x[:1])
    assert "script_name" in out and "top_ids" in out

check("models", test_models)

# ── 6. Training loop (1 epoch) ────────────────────────────────────────────────

def test_training():
    from transforms import get_transforms
    from datasets import ScriptImageDataset, make_loader, build_router_loaders
    from models import ScriptRouter, ScriptCNN
    from trainer import train_epoch, validate, train_model
    from utils import wrap_model
    from config import DEVICE

    # Router
    tr_dl, vl_dl, cw = build_router_loaders(
        deva_train  = DEVA_ROOT / "Train",
        tamil_train = TAMIL_ROOT / "train",
        bengali_dir = BENGALI_ROOT,
        telugu_csv  = TELUGU_CSV,
        max_per_script = 30,
    )
    model = wrap_model(ScriptRouter(4))
    hist, best = train_model(model, tr_dl, vl_dl, "smoke_router",
                              epochs=1, patience=1, class_weights=cw)
    assert "tr_loss" in hist

    # Character classifier
    ds_tr = ScriptImageDataset(DEVA_ROOT / "Train", transform=get_transforms("train"),
                                max_per_class=5)
    ds_vl = ScriptImageDataset(DEVA_ROOT / "Test",  transform=get_transforms("val"))
    dl_tr = make_loader(ds_tr, batch_size=4, shuffle=True)
    dl_vl = make_loader(ds_vl, batch_size=4, shuffle=False)

    model2 = wrap_model(ScriptCNN(N_CLS["deva"], 4))
    hist2, best2 = train_model(model2, dl_tr, dl_vl, "smoke_deva",
                                epochs=1, patience=1)
    assert "tr_loss" in hist2

check("training loop", test_training)

# ── 7. Evaluation ─────────────────────────────────────────────────────────────

def test_evaluation():
    from transforms import get_transforms
    from datasets import ScriptImageDataset, make_loader
    from models import ScriptCNN
    from trainer import load_checkpoint
    from evaluate import topk_acc, eval_script, plot_confusion
    from config import DEVICE, CKPT_DIR

    ckpt = CKPT_DIR / "smoke_deva_best.pth"
    if not ckpt.exists():
        print("  (skip eval: no checkpoint yet)")
        return

    ds = ScriptImageDataset(DEVA_ROOT / "Test", transform=get_transforms("val"))
    dl = make_loader(ds, batch_size=4, shuffle=False)
    m  = ScriptCNN(N_CLS["deva"], 4).to(DEVICE)
    load_checkpoint(m, ckpt)
    acc = topk_acc(m, dl, k=min(3, N_CLS["deva"]))
    assert 0 <= acc <= 100

    cfg = {
        "name"       : "smoke_deva",
        "num_classes": N_CLS["deva"],
        "num_layers" : 4,
        "dropout"    : 0.5,
        "batch"      : 4,
        "test_fn"    : lambda: ScriptImageDataset(
            DEVA_ROOT / "Test", transform=get_transforms("val")),
    }
    # Rename checkpoint so eval_script finds it
    ckpt_actual = CKPT_DIR / "smoke_deva_best.pth"
    ckpt_alias  = CKPT_DIR / "smoke_deva_best.pth"   # same path — no rename needed

    # eval_script looks for f"{name}_best.pth" i.e. smoke_deva_best.pth  ✓
    metrics, preds, lbls = eval_script("smoke_deva", cfg)
    if metrics:
        plot_confusion(preds, lbls, "smoke_deva")

check("evaluation", test_evaluation)

# ── 8. Ablation (1 epoch) ─────────────────────────────────────────────────────

def test_ablations():
    from ablations import run_all_ablations, plot_ablation_results
    results = run_all_ablations(
        deva_train    = DEVA_ROOT / "Train",
        deva_test     = DEVA_ROOT / "Test",
        max_per_class = 5,
    )
    assert isinstance(results, dict)
    plot_ablation_results(results)

check("ablations", test_ablations)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
if not FAILURES:
    print("SMOKE TEST PASSED — all checks OK")
else:
    print(f"SMOKE TEST FAILED — {len(FAILURES)} check(s) failed:")
    for f in FAILURES:
        print(f"  × {f}")

# Cleanup temp dir
shutil.rmtree(TMP, ignore_errors=True)

sys.exit(0 if not FAILURES else 1)
