import csv as _csv_mod
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from config import (CKPT_DIR, DEVICE, FIG_DIR, LOG_DIR, LR, PATIENCE,
                    WEIGHT_DECAY)
from utils import fmt_time, unwrap

# non_blocking transfers improve CUDA throughput by overlapping data movement
# with computation. On MPS (Apple Silicon) they cause crashes and must be off.
_NB = DEVICE.type == "cuda"


def _amp_supported() -> bool:
    """Probe whether float16 AMP works on the current CUDA device.

    Some GPU/driver combinations raise cudaErrorNoKernelImageForDevice on the
    first fp16 kernel launch even though CUDA is nominally available.  Running
    a tiny conv2d under autocast (with explicit synchronise to surface any
    asynchronous CUDA errors) catches the failure before training starts so
    the training loop can fall back to fp32 without crashing.
    """
    if DEVICE.type != "cuda":
        return False
    try:
        _x = torch.zeros(1, 1, 3, 3, device=DEVICE)
        _w = torch.zeros(1, 1, 1, 1, device=DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            torch.nn.functional.conv2d(_x, _w)
        torch.cuda.synchronize()   # surface any asynchronous CUDA errors now
        return True
    except Exception:
        return False


_USE_AMP: bool = _amp_supported()
import sys as _sys
print(
    f"[trainer] device={DEVICE} amp={_USE_AMP} "
    f"torch={torch.__version__} cuda_rt={torch.version.cuda}",
    file=_sys.stderr,
)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_acc: float,
    path,
) -> None:
    torch.save(
        {
            "epoch"     : epoch,
            "state_dict": unwrap(model).state_dict(),
            "opt_state" : optimizer.state_dict(),
            "val_acc"   : val_acc,
        },
        path,
    )


def load_checkpoint(model: nn.Module, path) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    unwrap(model).load_state_dict(ckpt["state_dict"])
    return ckpt["epoch"], ckpt["val_acc"]


class CSVLogger:
    def __init__(self, path, fields):
        self._fh = open(path, "w", newline="")
        self._wr = _csv_mod.DictWriter(self._fh, fieldnames=fields)
        self._wr.writeheader()

    def log(self, row: dict) -> None:
        self._wr.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler=None,
) -> Tuple[float, float]:
    model.train()
    total_loss = correct = total = 0

    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE, non_blocking=_NB)
        lbls = lbls.to(DEVICE, non_blocking=_NB)
        optimizer.zero_grad(set_to_none=True)

        # AMP is enabled only when _USE_AMP is True (probed at module load).
        # MPS and CPU always use fp32; some CUDA GPUs also fall back to fp32
        # if float16 kernels are not compiled for the assigned SM architecture.
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out  = model(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(lbls).sum().item()
        total      += imgs.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total


def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
) -> Tuple[float, float, list, list]:
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_lbls = [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE, non_blocking=_NB)
            lbls = lbls.to(DEVICE, non_blocking=_NB)

            if _USE_AMP:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out  = model(imgs)
                    loss = criterion(out, lbls)
            else:
                out  = model(imgs)
                loss = criterion(out, lbls)

            total_loss += loss.item() * imgs.size(0)
            preds       = out.argmax(1)
            correct    += preds.eq(lbls).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_lbls.extend(lbls.cpu().tolist())

    if total == 0:
        return 0.0, 0.0, [], []
    return total_loss / total, 100.0 * correct / total, all_preds, all_lbls


def train_model(
    model: nn.Module,
    tr_dl,
    vl_dl,
    tag: str,
    epochs: int,
    patience: int = PATIENCE,
    class_weights: Optional[torch.FloatTensor] = None,
) -> Tuple[dict, float]:
    ckpt_path = str(CKPT_DIR / f"{tag}_best.pth")
    # Class weights are injected into CrossEntropyLoss to counteract severe
    # class imbalance across scripts (e.g., Devanagari has 46 classes while
    # Telugu has only 6).
    criterion = (
        nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
        if class_weights is not None
        else nn.CrossEntropyLoss()
    )
    optimizer = optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    scaler = (
        torch.amp.GradScaler("cuda") if _USE_AMP else None
    )
    logger  = CSVLogger(
        str(LOG_DIR / f"{tag}.csv"),
        ["epoch", "tr_loss", "tr_acc", "vl_loss", "vl_acc", "lr"],
    )
    history  = defaultdict(list)
    best_acc = 0.0
    no_impr  = 0
    t0       = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc          = train_epoch(model, tr_dl, optimizer, criterion, scaler)
        vl_loss, vl_acc, _, _   = validate(model, vl_dl, criterion)
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        for k, v in [
            ("tr_loss", tr_loss), ("tr_acc", tr_acc),
            ("vl_loss", vl_loss), ("vl_acc", vl_acc),
        ]:
            history[k].append(v)

        logger.log(
            {
                "epoch"  : epoch,
                "tr_loss": f"{tr_loss:.4f}",
                "tr_acc" : f"{tr_acc:.2f}",
                "vl_loss": f"{vl_loss:.4f}",
                "vl_acc" : f"{vl_acc:.2f}",
                "lr"     : f"{cur_lr:.6f}",
            }
        )

        flag = ""
        if vl_acc > best_acc:
            best_acc = vl_acc
            no_impr  = 0
            save_checkpoint(model, optimizer, epoch, vl_acc, ckpt_path)
            flag = " [saved]"
        else:
            no_impr += 1

        if epoch % 5 == 0 or epoch == 1 or flag:
            print(
                f"  [{tag}] {epoch:3d}/{epochs}  "
                f"tr={tr_acc:.1f}%  vl={vl_acc:.1f}%  "
                f"best={best_acc:.1f}%  {fmt_time(time.time() - t0)}{flag}"
            )

        # Early stopping halts training when validation accuracy has not
        # improved for the allowed number of consecutive epochs.
        if no_impr >= patience:
            print(f"  Early stop at epoch {epoch} (no improvement for {patience} epochs)")
            break

    logger.close()
    print(f"  [{tag}] best val acc = {best_acc:.2f}%\n")
    return dict(history), best_acc


def plot_history(history: dict, tag: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    ax1.plot(history["tr_loss"], label="train")
    ax1.plot(history["vl_loss"], label="val")
    ax1.set(title=f"{tag} Loss", xlabel="epoch", ylabel="loss")
    ax1.legend()

    ax2.plot(history["tr_acc"], label="train")
    ax2.plot(history["vl_acc"], label="val")
    ax2.set(title=f"{tag} Accuracy", xlabel="epoch", ylabel="%")
    ax2.legend()

    plt.tight_layout()
    out = FIG_DIR / f"curve_{tag}.png"
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  Saved curve → {out}")
