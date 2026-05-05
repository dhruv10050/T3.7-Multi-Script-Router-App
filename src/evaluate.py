import timeit
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision import transforms

from config import (CKPT_DIR, DEVICE, FIG_DIR, IMG_SIZE, NORM_MEAN, NORM_STD)
from datasets import make_loader
from models import ScriptCNN, ScriptRouter
from trainer import load_checkpoint, validate
from transforms import get_transforms
from utils import free_memory, unwrap


def topk_acc(model: nn.Module, loader, k: int = 5) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE, non_blocking=(DEVICE.type=="cuda"))
            lbls = lbls.to(DEVICE, non_blocking=(DEVICE.type=="cuda"))
            topk = model(imgs).topk(k, dim=1).indices
            correct += topk.eq(lbls.view(-1, 1)).any(dim=1).sum().item()
            total   += imgs.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def eval_script(
    name: str, cfg: dict
) -> Tuple[Optional[dict], Optional[list], Optional[list]]:
    """
    Loads the best checkpoint for a given script, evaluates it on the test set,
    and returns metrics alongside raw predictions and ground-truth labels.
    Returns (None, None, None) when the checkpoint or test data is unavailable.
    """
    ckpt = CKPT_DIR / f"{name}_best.pth"
    if not ckpt.exists():
        print(f"  SKIP {name}: no checkpoint at {ckpt}")
        return None, None, None

    te_ds = cfg["test_fn"]()
    if te_ds is None:
        print(f"  SKIP {name}: test dataset unavailable")
        return None, None, None

    te_dl = make_loader(te_ds, cfg["batch"], shuffle=False)
    model = ScriptCNN(
        cfg["num_classes"], cfg["num_layers"], cfg["dropout"]
    ).to(DEVICE)
    load_checkpoint(model, ckpt)
    model.eval()

    crit                       = nn.CrossEntropyLoss()
    _, top1, preds, lbls       = validate(model, te_dl, crit)
    top5                       = topk_acc(model, te_dl, k=min(5, cfg["num_classes"]))
    mf1                        = f1_score(lbls, preds, average="macro",    zero_division=0) * 100
    wf1                        = f1_score(lbls, preds, average="weighted", zero_division=0) * 100

    metrics = {
        "script"     : name,
        "num_classes": cfg["num_classes"],
        "test_size"  : len(te_ds),
        "top1_acc"   : round(top1, 2),
        "top5_acc"   : round(top5, 2),
        "macro_f1"   : round(mf1,  2),
        "weighted_f1": round(wf1,  2),
    }
    del model, te_ds, te_dl
    free_memory()
    return metrics, preds, lbls


def plot_confusion(
    preds: list, lbls: list, name: str, max_cls: int = 40
) -> None:
    # Display is capped at max_cls classes to keep the matrix legible for
    # scripts with large character inventories such as Devanagari (46 classes).
    cm   = confusion_matrix(lbls, preds)
    n    = cm.shape[0]
    show = min(n, max_cls)
    h    = max(6,  show * 0.28)
    w    = max(8,  show * 0.30)

    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(cm[:show, :show], cmap="Blues", ax=ax,
                cbar=True, linewidths=0.3)
    ax.set(
        title=f"{name} Confusion Matrix" + (f" (first {show})" if show < n else ""),
        xlabel="Predicted",
        ylabel="True",
    )
    plt.tight_layout()
    out = FIG_DIR / f"cm_{name}.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved confusion matrix → {out}")


def _perturb_tfm(extra: list) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ]
        + extra
        + [
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


# All perturbations use fixed parameters so robustness tables are exactly
# reproducible across runs without requiring a separate global random seed.
PERTURBATIONS = {
    "clean"      : _perturb_tfm([]),
    "blur_light" : _perturb_tfm([transforms.GaussianBlur(3, sigma=1.0)]),
    "blur_heavy" : _perturb_tfm([transforms.GaussianBlur(5, sigma=2.0)]),
    "rotate_10"  : _perturb_tfm([transforms.functional.rotate
                                   if False else transforms.RandomRotation(
                                       (10, 10))]),  # fixed +10 deg
    "rotate_20"  : _perturb_tfm([transforms.RandomRotation((20, 20))]),   # fixed +20 deg
    "brightness" : _perturb_tfm([transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 1.4))]),
    "crop_pad6"  : _perturb_tfm([transforms.Pad(6), transforms.CenterCrop(IMG_SIZE)]),
}


def run_robustness_eval(script_cfgs: list) -> dict:
    """
    Evaluates each script model under each perturbation.
    Returns {script_name: {perturbation: accuracy}}.
    """
    results = {}
    crit    = nn.CrossEntropyLoss()

    for cfg in script_cfgs:
        name = cfg["name"]
        ckpt = CKPT_DIR / f"{name}_best.pth"
        if not ckpt.exists():
            continue
        te_base = cfg["test_fn"]()
        if te_base is None:
            continue

        model = ScriptCNN(
            cfg["num_classes"], cfg["num_layers"], cfg["dropout"]
        ).to(DEVICE)
        load_checkpoint(model, ckpt)
        model.eval()
        results[name] = {}

        for label, tfm in PERTURBATIONS.items():
            te_p           = cfg["test_fn"]()
            te_p.transform = tfm
            dl             = make_loader(te_p, cfg["batch"], shuffle=False)
            _, acc, _, _   = validate(model, dl, crit)
            results[name][label] = round(acc, 2)
            del te_p, dl

        del model, te_base
        free_memory()
        print(f"  {name}: {results[name]}")

    return results


def plot_robustness(results: dict) -> None:
    if not results:
        return
    rob_df = pd.DataFrame(results).T
    print("\nRobustness Accuracy (%)")
    print(rob_df.to_string())

    fig, ax = plt.subplots(figsize=(11, 4))
    rob_df.T.plot(kind="bar", ax=ax, width=0.7)
    ax.set(title="Robustness Evaluation", ylabel="Accuracy (%)", xlabel="Perturbation")
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = FIG_DIR / "robustness.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved robustness plot → {out}")


def _benchmark_model(model: nn.Module, n_runs: int = 200) -> Tuple[float, float]:
    """Measures CPU latency in milliseconds for a single-image forward pass."""
    m = unwrap(model).cpu().eval()
    d = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
    # A warm-up phase of 20 runs allows JIT and cache effects to stabilise
    # before timing begins.
    for _ in range(20):
        m(d)
    ts = timeit.repeat(lambda: m(d), number=1, repeat=n_runs)
    ms = np.array(ts) * 1000
    return float(ms.mean()), float(ms.std())


def run_latency_benchmark(script_cfgs: list, n_runs: int = 200) -> dict:
    """Benchmarks all script classifiers plus the router on CPU.

    Accumulates router latency plus classifier latency to report the
    practical end-to-end deployment cost for each script path.
    Returns {model_name: {"mean_ms": ..., "std_ms": ...}}.
    """
    results = {}
    print(f"Latency benchmark — CPU, 1 image, {n_runs} runs")
    print(f"{'Model':<24} {'Mean ms':>10} {'Std ms':>8}")
    print("-" * 46)

    rtr = ScriptRouter()
    r_ckpt = CKPT_DIR / "router_best.pth"
    if r_ckpt.exists():
        load_checkpoint(rtr, r_ckpt)
    mu, sd = _benchmark_model(rtr, n_runs)
    results["router"] = {"mean_ms": round(mu, 2), "std_ms": round(sd, 2)}
    print(f"  {'router':<22} {mu:>9.2f}  {sd:>7.2f}")

    router_ms = results.get("router", {}).get("mean_ms", 0)
    for cfg in script_cfgs:
        ckpt = CKPT_DIR / f"{cfg['name']}_best.pth"
        if not ckpt.exists():
            continue
        m = ScriptCNN(cfg["num_classes"], cfg["num_layers"], cfg["dropout"])
        load_checkpoint(m, ckpt)
        mu, sd = _benchmark_model(m, n_runs)
        results[cfg["name"]] = {"mean_ms": round(mu, 2), "std_ms": round(sd, 2)}
        e2e = round(router_ms + mu, 2)
        results[cfg["name"]]["e2e_ms"] = e2e
        print(f"  {cfg['name']:<22} {mu:>9.2f}  {sd:>7.2f}   e2e={e2e:.2f} ms")

    # Worst-case end-to-end across all scripts
    script_e2e = [v["e2e_ms"] for k, v in results.items() if "e2e_ms" in v]
    worst_e2e  = max(script_e2e) if script_e2e else router_ms
    results["end_to_end_worst"] = {"mean_ms": round(worst_e2e, 2), "std_ms": 0}
    print("-" * 46)
    print(f"  {'End-to-end worst-case':<22} {worst_e2e:>9.2f} ms")
    return results
