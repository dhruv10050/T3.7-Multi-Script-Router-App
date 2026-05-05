import random
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import (CHAR_BATCH, IMG_SIZE, NUM_WORKERS, PIN_MEMORY,
                    PERSISTENT_W, ROUTER_BATCH, ROUTER_MAX, SEED,
                    TELUGU_BATCH, WORK_DIR)
from transforms import get_transforms


class ScriptImageDataset(Dataset):
    """Image-folder dataset with optional per-class sample cap.

    When split is 'train' or 'val', a stratified split is carved from
    the root folder itself rather than requiring a separate validation
    directory.  This allows datasets that ship with a single Train/ folder
    (e.g. Devanagari, Tamil) to produce a held-out validation set without
    duplicating data on disk."""

    def __init__(
        self,
        root,
        transform=None,
        max_per_class: Optional[int] = None,
        seed: int = SEED,
        split: Optional[str] = None,
        val_ratio: float = 0.15,
    ):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: dict = {}

        root = Path(root)
        for idx, cd in enumerate(
            sorted(d for d in root.iterdir() if d.is_dir())
        ):
            self.classes.append(cd.name)
            self.class_to_idx[cd.name] = idx
            imgs = (
                list(cd.glob("*.png"))
                + list(cd.glob("*.jpg"))
                + list(cd.glob("*.jpeg"))
                + list(cd.glob("*.bmp"))
            )
            if max_per_class and len(imgs) > max_per_class:
                imgs = random.Random(seed).sample(imgs, max_per_class)
            if split in ("train", "val") and len(imgs) >= 2:
                tr_imgs, va_imgs = train_test_split(
                    imgs, test_size=val_ratio, random_state=seed
                )
                imgs = va_imgs if split == "val" else tr_imgs
            self.samples.extend((str(p), idx) for p in imgs)

        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("L")
        except Exception:
            img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_weights(self) -> torch.FloatTensor:
        counts = np.zeros(self.num_classes, dtype=np.float32)
        for _, lbl in self.samples:
            counts[lbl] += 1
        w = 1.0 / (counts + 1e-6)
        return torch.FloatTensor(w / w.sum() * self.num_classes)


class BanglaLekhaDataset(Dataset):
    """Bengali BanglaLekha-Isolated dataset with stratified 70/15/15 splits.

    The full dataset is partitioned at construction time so that train,
    val, and test subsets share a consistent random state and never overlap.
    A two-pass stratified split preserves class proportions across all three
    partitions."""

    def __init__(
        self,
        root,
        split: str = "train",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        transform=None,
        max_per_class: Optional[int] = None,
        seed: int = SEED,
    ):
        self.transform = transform
        root = Path(root)
        class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        self.num_classes  = len(class_dirs)
        self.classes      = [d.name for d in class_dirs]

        all_paths, all_lbls = [], []
        for idx, cd in enumerate(class_dirs):
            imgs = sorted(
                list(cd.glob("*.png")) + list(cd.glob("*.jpg"))
            )
            all_paths.extend(map(str, imgs))
            all_lbls.extend([idx] * len(imgs))

        all_paths = np.array(all_paths)
        all_lbls  = np.array(all_lbls, dtype=np.int64)
        idx_all   = np.arange(len(all_paths))

        tr_i, tmp = train_test_split(
            idx_all,
            test_size=val_ratio + test_ratio,
            stratify=all_lbls,
            random_state=seed,
        )
        vl_i, te_i = train_test_split(
            tmp,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=all_lbls[tmp],
            random_state=seed,
        )
        sel = {"train": tr_i, "val": vl_i, "test": te_i}[split]
        self.paths  = all_paths[sel]
        self.labels = all_lbls[sel]

        if max_per_class:
            rng  = np.random.RandomState(seed)
            keep = []
            for c in range(self.num_classes):
                ci = np.where(self.labels == c)[0]
                if len(ci) > max_per_class:
                    ci = rng.choice(ci, max_per_class, replace=False)
                keep.extend(ci.tolist())
            self.paths  = self.paths[keep]
            self.labels = self.labels[keep]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        try:
            img = Image.open(self.paths[i]).convert("L")
        except Exception:
            img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[i])

    def class_weights(self) -> torch.FloatTensor:
        counts = np.bincount(
            self.labels, minlength=self.num_classes
        ).astype(np.float32)
        w = 1.0 / (counts + 1e-6)
        return torch.FloatTensor(w / w.sum() * self.num_classes)


class TeluguCSVDataset(Dataset):
    """Telugu 6-vowel dataset stored as pixel rows in a CSV file.

    Each row encodes one flattened grayscale image; the last column holds
    the class label.  Pixel values are normalised to uint8 and reshaped into
    a square image before being served to the transform pipeline."""

    def __init__(
        self,
        csv_path,
        split: str = "train",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        transform=None,
        seed: int = SEED,
    ):
        self.transform = transform
        df        = pd.read_csv(csv_path)
        label_col = df.columns[-1]
        pixels    = df.iloc[:, :-1].values.astype(np.float32)
        lbl_raw   = df[label_col].values

        unique  = sorted(np.unique(lbl_raw))
        lbl_map = {l: i for i, l in enumerate(unique)}
        labels  = np.array([lbl_map[l] for l in lbl_raw], dtype=np.int64)

        self.idx_to_label = {i: l for l, i in lbl_map.items()}
        self.num_classes  = len(unique)
        self.classes      = [str(u) for u in unique]

        vmax = pixels.max()
        px8  = (pixels / (vmax if vmax > 1.0 else 1.0) * 255).astype(np.uint8)
        side = int(np.sqrt(px8.shape[1]))
        self._pixels = px8.reshape(-1, side, side)

        idx_all = np.arange(len(labels))
        tr_i, tmp = train_test_split(
            idx_all,
            test_size=val_ratio + test_ratio,
            stratify=labels,
            random_state=seed,
        )
        vl_i, te_i = train_test_split(
            tmp,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=labels[tmp],
            random_state=seed,
        )
        sel         = {"train": tr_i, "val": vl_i, "test": te_i}[split]
        self.data   = self._pixels[sel]
        self.labels = labels[sel]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = Image.fromarray(self.data[i], mode="L")
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[i])

    def class_weights(self) -> torch.FloatTensor:
        counts = np.bincount(
            self.labels, minlength=self.num_classes
        ).astype(np.float32)
        w = 1.0 / (counts + 1e-6)
        return torch.FloatTensor(w / w.sum() * self.num_classes)


class ListDataset(Dataset):
    """Generic dataset built from an explicit list of (path_or_array, label)
    pairs.  Used by the router to assemble a cross-script training set from
    heterogeneous per-script sources without requiring a shared directory
    structure."""

    def __init__(self, samples: list, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        src, label = self.samples[i]
        if isinstance(src, np.ndarray):
            img = Image.fromarray(src, mode="L")
        else:
            try:
                img = Image.open(src).convert("L")
            except Exception:
                img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
        if self.transform:
            img = self.transform(img)
        return img, label


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    sampler=None,
) -> DataLoader:
    # drop_last is activated only when shuffling and the dataset is large
    # enough to fill at least one complete batch, preventing a noisy partial
    # batch from disproportionately influencing gradient updates.
    drop_last = shuffle and len(dataset) >= batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_W,
        drop_last=drop_last,
    )


def weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    if len(dataset) == 0:
        raise ValueError("weighted_sampler received an empty dataset")
    # Label metadata is read directly from dataset attributes when available
    # to avoid triggering image decoding across the entire dataset.
    if hasattr(dataset, "labels"):
        lbls = dataset.labels.tolist() if hasattr(dataset.labels, "tolist") else list(dataset.labels)
    elif hasattr(dataset, "samples"):
        lbls = [lbl for _, lbl in dataset.samples]
    else:
        lbls = [dataset[i][1] for i in range(len(dataset))]
    lbls_arr = np.asarray(lbls, dtype=np.int64)
    counts = np.bincount(lbls_arr)
    wts    = 1.0 / (counts[lbls_arr] + 1e-6)
    return WeightedRandomSampler(
        torch.FloatTensor(wts), len(wts), replacement=True
    )


def _collect_img_samples(
    root, script_id: int, max_n: int, seed: int = SEED
) -> list:
    imgs = (list(Path(root).rglob("*.png")) + list(Path(root).rglob("*.jpg"))
            + list(Path(root).rglob("*.jpeg")) + list(Path(root).rglob("*.bmp")))
    if len(imgs) > max_n:
        imgs = random.Random(seed).sample(imgs, max_n)
    return [(str(p), script_id) for p in imgs]


def _collect_telugu_samples(
    csv_path, script_id: int, max_n: int, seed: int = SEED
) -> list:
    df   = pd.read_csv(csv_path)
    px   = df.iloc[:, :-1].values.astype(np.float32)
    vmax = px.max()
    u8   = (px / (vmax if vmax > 1 else 1.0) * 255).astype(np.uint8)
    side = int(np.sqrt(u8.shape[1]))
    arrs = [u8[i].reshape(side, side) for i in range(len(u8))]
    if len(arrs) > max_n:
        arrs = random.Random(seed).sample(arrs, max_n)
    return [(arr, script_id) for arr in arrs]


def build_router_loaders(
    deva_train,
    tamil_train,
    bengali_dir,
    telugu_csv,
    max_per_script: int = ROUTER_MAX,
    val_ratio: float = 0.15,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, torch.FloatTensor]:
    """Assembles the 4-class router dataset from all four script sources.

    Script label assignment: 0=Devanagari, 1=Tamil, 2=Bengali, 3=Telugu.
    A WeightedRandomSampler is applied to the training loader to compensate
    for Telugu's smaller dataset size relative to the other scripts."""
    all_samples = []
    if deva_train:
        all_samples += _collect_img_samples(deva_train,  0, max_per_script, seed)
    if tamil_train:
        all_samples += _collect_img_samples(tamil_train, 1, max_per_script, seed)
    if bengali_dir:
        all_samples += _collect_img_samples(bengali_dir, 2, max_per_script, seed)
    if telugu_csv:
        all_samples += _collect_telugu_samples(telugu_csv, 3, max_per_script, seed)

    if not all_samples:
        raise ValueError("No data found for router — check dataset paths.")

    random.Random(seed).shuffle(all_samples)
    lbls    = [s[1] for s in all_samples]
    counts  = np.bincount(lbls, minlength=4).astype(np.float32)
    w       = 1.0 / (counts + 1e-6)
    class_w = torch.FloatTensor(w / w.sum() * 4)

    idx_all = list(range(len(all_samples)))
    tr_i, vl_i = train_test_split(
        idx_all, test_size=val_ratio, stratify=lbls, random_state=seed
    )
    tr_ds = ListDataset(
        [all_samples[i] for i in tr_i], transform=get_transforms("train")
    )
    vl_ds = ListDataset(
        [all_samples[i] for i in vl_i], transform=get_transforms("val")
    )
    tr_lbls = [all_samples[i][1] for i in tr_i]
    tr_wts  = 1.0 / (counts[tr_lbls] + 1e-6)
    sampler = WeightedRandomSampler(
        torch.FloatTensor(tr_wts), len(tr_i), replacement=True
    )
    tr_dl = make_loader(tr_ds, ROUTER_BATCH, shuffle=False, sampler=sampler)
    vl_dl = make_loader(vl_ds, ROUTER_BATCH, shuffle=False)
    return tr_dl, vl_dl, class_w
import random
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import (CHAR_BATCH, IMG_SIZE, NUM_WORKERS, PIN_MEMORY,
                    PERSISTENT_W, ROUTER_BATCH, ROUTER_MAX, SEED,
                    TELUGU_BATCH, WORK_DIR)
from transforms import get_transforms

