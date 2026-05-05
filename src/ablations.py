import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from config import (ABLATION_EPOCHS, CHAR_BATCH, CKPT_DIR, DEVICE,
                    FIG_DIR, LR, PATIENCE, SEED, WEIGHT_DECAY)
from datasets import ScriptImageDataset, make_loader
from models import ScriptCNN
from trainer import save_checkpoint, train_epoch, validate
from transforms import get_transforms
from utils import free_memory, unwrap, wrap_model


def ablation_train(
    tag: str,
    model_fn,
    tr_dl,
    vl_dl,
    epochs: int = ABLATION_EPOCHS,
) -> float:
    """Trains one ablation variant with CosineAnnealingLR and early stopping
    consistent with the main pipeline so that results are directly comparable.
    Returns the best validation accuracy (%)."""
    model   = wrap_model(model_fn())
    crit    = nn.CrossEntropyLoss()
    opt     = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler  = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    ckpt    = str(CKPT_DIR / f"abl_{tag}.pth")
    best    = 0.0
    no_impr = 0
    pat     = max(2, PATIENCE - 1)

    for ep in range(1, epochs + 1):
        train_epoch(model, tr_dl, opt, crit, scaler)
        _, va, _, _ = validate(model, vl_dl, crit)
        sched.step()
        if va > best:
            best    = va
            no_impr = 0
            save_checkpoint(model, opt, ep, va, ckpt)
        else:
            no_impr += 1
        if no_impr >= pat:
            break

    del model
    free_memory()
    return round(best, 2)


def run_study(
    study_name: str, variants: list, tr_dl, vl_dl, results: dict
) -> None:
    print(f"\n  ── {study_name} ──")
    results[study_name] = {}
    for label, fn in variants:
        acc = ablation_train(f"{study_name}_{label}", fn, tr_dl, vl_dl)
        results[study_name][label] = acc
        print(f"    {label:<40} {acc:>6.2f}%")



def run_all_ablations(
    deva_train,
    deva_test,
    max_per_class: Optional[int] = None,
) -> dict:
    """Runs all 7 ablation studies on Devanagari and returns a results dict.

    Devanagari is used because it has the largest character inventory
    (46 classes), making ablation differences more statistically meaningful.
    """
    if deva_train is None or deva_test is None:
        print("Ablations require Devanagari dataset — skipping.")
        return {}

    NC = 46
    results: dict = {}

    def _make_dl(root, mode, res=None, max_pc=None):
        ds  = ScriptImageDataset(
            root,
            transform=get_transforms(mode, res or 64),
            max_per_class=max_pc or max_per_class,
        )
        return make_loader(ds, CHAR_BATCH, shuffle=(mode == "train"))

    tr_dl = _make_dl(deva_train, "train")
    vl_dl = _make_dl(deva_test,  "val")

    print("=" * 60)
    print("ABLATION STUDIES — Devanagari (46 classes)")
    print("=" * 60)


    run_study("1_cnn_depth", [
        ("2_layers",           lambda: ScriptCNN(NC, 2)),
        ("3_layers",           lambda: ScriptCNN(NC, 3)),
        ("4_layers_default",   lambda: ScriptCNN(NC, 4)),
    ], tr_dl, vl_dl, results)

    # Study 2 rebuilds DataLoaders at each resolution because the model's
    # fully-connected head dimensions change with spatial size.
    print("\n  ── 2_input_resolution ──")
    results["2_input_resolution"] = {}
    for res in [28, 32, 64]:
        _tr = _make_dl(deva_train, "train", res=res)
        _vl = _make_dl(deva_test,  "val",   res=res)
        def _model_fn(r=res):
            hw     = r // (2 ** 4)
            fc_in  = 256 * hw * hw
            fc_hid = max(256, fc_in // 8)
            m      = ScriptCNN(NC, 4)
            m.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(fc_in, fc_hid), nn.ReLU(inplace=True),
                nn.Dropout(0.5), nn.Linear(fc_hid, NC),
            )
            return m
        acc = ablation_train(f"2_res{res}", _model_fn, _tr, _vl)
        results["2_input_resolution"][f"{res}x{res}"] = acc
        print(f"    {res}x{res:<38} {acc:>6.2f}%")
        del _tr, _vl
        free_memory()

    def _no_bn():
        m = ScriptCNN(NC, 4)
        m.features = nn.Sequential(
            *[l for l in m.features if not isinstance(l, nn.BatchNorm2d)]
        )
        return m

    run_study("3_batchnorm", [
        ("no_batchnorm",            _no_bn),
        ("with_batchnorm_default",  lambda: ScriptCNN(NC, 4)),
    ], tr_dl, vl_dl, results)


    run_study("4_dropout", [
        ("p_0.0", lambda: ScriptCNN(NC, 4, dropout=0.0)),
        ("p_0.3", lambda: ScriptCNN(NC, 4, dropout=0.3)),
        ("p_0.5_default", lambda: ScriptCNN(NC, 4, dropout=0.5)),
        ("p_0.7", lambda: ScriptCNN(NC, 4, dropout=0.7)),
    ], tr_dl, vl_dl, results)

    # Study 5 uses the validation transform as the no-augmentation baseline
    # so the comparison is controlled for all other factors.
    _no_aug_dl = _make_dl(deva_train, "val")
    acc_no  = ablation_train("5_no_aug",   lambda: ScriptCNN(NC, 4), _no_aug_dl, vl_dl)
    acc_yes = ablation_train("5_full_aug", lambda: ScriptCNN(NC, 4), tr_dl,      vl_dl)
    results["5_augmentation"] = {
        "no_augmentation"           : acc_no,
        "full_augmentation_default" : acc_yes,
    }
    print("\n  ── 5_augmentation ──")
    print(f"    {'no_augmentation':<40} {acc_no:>6.2f}%")
    print(f"    {'full_augmentation_default':<40} {acc_yes:>6.2f}%")
    del _no_aug_dl
    free_memory()

    # Study 6: ReduceLROnPlateau requires the validation metric rather than
    # epoch count when stepping, so it is handled separately from the others.
    print("\n  ── 6_lr_scheduler ──")
    results["6_lr_scheduler"] = {}
    for sn in ["none", "StepLR", "CosineAnn", "ReduceLROnPlateau"]:
        ep    = ABLATION_EPOCHS
        model = wrap_model(ScriptCNN(NC, 4))
        opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        if   sn == "none":               sched_obj = None
        elif sn == "StepLR":             sched_obj = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
        elif sn == "CosineAnn":          sched_obj = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
        else:                            sched_obj = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
        crit   = nn.CrossEntropyLoss()
        sc     = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
        best   = 0.0; no_i = 0; pat = max(2, PATIENCE - 1)
        for e in range(1, ep + 1):
            train_epoch(model, tr_dl, opt, crit, sc)
            _, va, _, _ = validate(model, vl_dl, crit)
            if sched_obj:
                (sched_obj.step(va)
                 if isinstance(sched_obj, optim.lr_scheduler.ReduceLROnPlateau)
                 else sched_obj.step())
            if va > best:
                best = va; no_i = 0
            else:
                no_i += 1
            if no_i >= pat:
                break
        del model
        free_memory()
        results["6_lr_scheduler"][sn] = round(best, 2)
        print(f"    {sn:<40} {best:>6.2f}%")

    # Study 7 uses CNN depth as a proxy for pipeline complexity. A true
    # two-stage vs. unified comparison requires cross-script data and is
    # tracked separately in the main evaluation.
    run_study("7_architecture_depth_proxy", [
        ("shallow_2layer",          lambda: ScriptCNN(NC, 2)),
        ("deep_4layer_default",      lambda: ScriptCNN(NC, 4)),
    ], tr_dl, vl_dl, results)

    del tr_dl, vl_dl
    free_memory()
    print("\nAll ablation studies complete.")
    return results


def plot_ablation_results(results: dict) -> None:
    if not results:
        print("No ablation results to plot.")
        return

    studies = list(results.items())
    nc = len(studies)
    fig, axes = plt.subplots(1, nc, figsize=(4.5 * nc, 4), sharey=False)
    if nc == 1:
        axes = [axes]

    for ax, (study, variants) in zip(axes, studies):
        labels = list(variants.keys())
        vals   = list(variants.values())
        colors = ["#FF5722" if "default" in l else "#2196F3" for l in labels]
        bars   = ax.barh(range(len(labels)), vals, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(
            [l.replace("_default", " *") for l in labels], fontsize=7
        )
        ax.set_title(study.replace("_", " "), fontsize=8, fontweight="bold")
        ax.set_xlabel("Val Acc (%)", fontsize=7)
        vmin, vmax = min(vals), max(vals)
        ax.set_xlim(max(0, vmin - 3), min(100, vmax + 3))
        for bar, v in zip(bars, vals):
            ax.text(
                v + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=7,
            )

    plt.suptitle(
        "Ablation Studies — Devanagari Val Accuracy  (* = default)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out = FIG_DIR / "ablations.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved ablation chart → {out}")

    print(f"\n{'Study':<28} {'Best variant':<42} {'Acc':>6}")
    print("-" * 78)
    for study, variants in results.items():
        bk = max(variants, key=variants.get)
        print(f"{study:<28} {bk:<42} {variants[bk]:>5.2f}%")
