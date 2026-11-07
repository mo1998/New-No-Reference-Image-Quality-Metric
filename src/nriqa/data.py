"""Dataset loaders, content-level splits and augmentation for NR-IQA.

The golden rule for this project is that a train/validation/test boundary is a
boundary between *reference contents*, never between distorted images.  Both
KADID-10k and TID2013 provide the reference filename for every distorted image,
so the rule can be enforced exactly.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ---------------------------------------------------------------------------
# Label / filename parsing
# ---------------------------------------------------------------------------
_KADID_RE = re.compile(r"^(?P<ref>I\d+)_(?P<dtype>\d+)_(?P<level>\d+)\.png$", re.I)
_TID_RE = re.compile(r"^(?P<ref>I\d+)_(?P<dtype>\d+)_(?P<level>\d+)\.bmp$", re.I)


def _kadid_rows(root: str) -> List[Dict[str, Any]]:
    csv_path = os.path.join(root, "dmos.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"KADID-10k label file not found: {csv_path}")
    labels = pd.read_csv(csv_path)
    img_dir = os.path.join(root, "images")
    rows: List[Dict[str, Any]] = []
    missing = 0
    for rec in labels.itertuples(index=False):
        name = str(rec.dist_img)
        m = _KADID_RE.match(name)
        if not m:
            raise ValueError(f"Unexpected KADID filename: {name}")
        path = os.path.join(img_dir, name)
        if not os.path.exists(path):
            missing += 1
            continue
        rows.append(
            {
                "dataset": "kadid10k",
                "name": name,
                "path": path,
                "ref": f"{m.group('ref').upper()}.png",
                "mos": float(rec.dmos),
                "mos_std": getattr(rec, "var", float("nan")),
                "dist_type": int(m.group("dtype")),
                "level": int(m.group("level")),
            }
        )
    if missing:
        raise FileNotFoundError(f"{missing} KADID images are missing under {img_dir}")
    return rows


def _tid_rows(root: str) -> List[Dict[str, Any]]:
    """Parse TID2013.

    TID2013 filenames are inconsistently cased (``I01_01_1.bmp`` vs
    ``i01_01_2.bmp``), so lookups are done through a lower-cased index.
    """
    label_path = os.path.join(root, "mos_with_names.txt")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"TID2013 label file not found: {label_path}")
    dist_dir = os.path.join(root, "distorted_images")
    ref_dir = os.path.join(root, "reference_images")

    def index_dir(d: str) -> Dict[str, str]:
        return {f.lower(): os.path.join(d, f) for f in os.listdir(d)}

    dist_index = index_dir(dist_dir)
    ref_index = index_dir(ref_dir)
    rows: List[Dict[str, Any]] = []
    with open(label_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            mos, name = float(parts[0]), parts[1]
            m = _TID_RE.match(name)
            if not m:
                raise ValueError(f"Unexpected TID2013 filename: {name}")
            key = name.lower()
            if key not in dist_index:
                raise FileNotFoundError(f"Missing TID2013 distorted image: {name}")
            rows.append(
                {
                    "dataset": "tid2013",
                    "name": name,
                    "path": dist_index[key],
                    "ref": f"{m.group('ref').upper()}.bmp",
                    "ref_path": ref_index[f"{m.group('ref').lower()}.bmp"],
                    "mos": mos,
                    "mos_std": np.nan,
                    "dist_type": int(m.group("dtype")),
                    "level": int(m.group("level")),
                }
            )
    return rows


def load_dataframe(dataset: str, root: str) -> pd.DataFrame:
    """Load a supported dataset into the canonical dataframe schema.

    Columns: dataset, name, path, ref, mos, mos_std, dist_type, level.
    ``mos`` is always oriented so that higher means better quality.
    """
    dataset = dataset.lower()
    if dataset in ("kadid10k", "kadid", "kadid-10k"):
        rows = _kadid_rows(root)
    elif dataset in ("tid2013", "tid", "tid-2013"):
        rows = _tid_rows(root)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    df = pd.DataFrame(rows)
    # Deterministic row order, independent of filesystem ordering
    df = df.sort_values(["ref", "dist_type", "level", "name"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Content-level splits
# ---------------------------------------------------------------------------
def _content_refs(refs: Iterable[str]) -> List[str]:
    return sorted({str(r) for r in refs})


def split_dataframe(
    df: pd.DataFrame,
    mode: str = "holdout",
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 42,
    fold: int = 0,
    kfold: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) dataframes split by reference content.

    ``val_frac`` is a fraction of *all* reference images.  In k-fold mode the
    test fold is one of ``kfold`` disjoint content folds and validation is
    carved out of the remaining contents.
    """
    refs = _content_refs(df["ref"])
    if len(refs) < 5:
        raise ValueError("Need at least 5 reference contents to split")
    rng = np.random.default_rng(seed)

    if mode == "holdout":
        perm = rng.permutation(len(refs))
        n_test = max(1, int(round(len(refs) * test_frac)))
        n_val = max(1, int(round(len(refs) * val_frac)))
        test_refs = sorted(refs[i] for i in perm[:n_test])
        val_refs = sorted(refs[i] for i in perm[n_test : n_test + n_val])
        train_refs = sorted(
            r for i, r in enumerate(refs) if r not in set(test_refs) | set(val_refs)
        )
    elif mode == "kfold":
        if not 0 <= fold < kfold:
            raise ValueError(f"fold must be in [0, {kfold}), got {fold}")
        perm = rng.permutation(len(refs))
        chunks = np.array_split(perm, kfold)
        test_refs = sorted(refs[i] for i in chunks[fold])
        remaining = [refs[i] for i in perm if refs[i] not in set(test_refs)]
        # Deterministic validation selection from the remaining contents
        n_val = max(1, int(round(len(refs) * val_frac)))
        val_refs = sorted(remaining[:n_val])
        train_refs = sorted(remaining[n_val:])
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    if not train_refs or not val_refs or not test_refs:
        raise ValueError(
            f"Empty split: train={len(train_refs)} val={len(val_refs)} test={len(test_refs)}"
        )
    assert not (set(train_refs) & set(val_refs))
    assert not (set(train_refs) & set(test_refs))
    assert not (set(val_refs) & set(test_refs))

    def take(names: Sequence[str]) -> pd.DataFrame:
        return df[df["ref"].isin(set(names))].copy()

    return take(train_refs), take(val_refs), take(test_refs)


def split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Any]:
    return {
        "train_images": int(len(train)),
        "val_images": int(len(val)),
        "test_images": int(len(test)),
        "train_refs": int(train["ref"].nunique()),
        "val_refs": int(val["ref"].nunique()),
        "test_refs": int(test["ref"].nunique()),
        "train_datasets": sorted(train["dataset"].unique().tolist()) if "dataset" in train else [],
    }


def load_extra_train(cfg: Dict[str, Any], exclude: Sequence[str] = ()) -> pd.DataFrame:
    """Load extra datasets for multi-dataset training.

    MOS scales differ across datasets (KADID 1-5, TID2013 0-9).  Each extra
    dataset is linearly mapped to the KADID-like [1, 5] interval so a single
    regression head is meaningful.  The mapping is monotonic and therefore
    does not change rank-based evaluation.
    """
    data_cfg = cfg["data"]
    extras = []
    for name in data_cfg.get("extra_datasets", []) or []:
        if name in exclude:
            continue
        if name.lower() in ("tid2013", "tid"):
            df = load_dataframe("tid2013", data_cfg.get("tid_root", "data/tid2013"))
        else:
            raise ValueError(f"Unsupported extra dataset: {name}")
        lo, hi = float(df["mos"].min()), float(df["mos"].max())
        if hi <= lo:
            raise ValueError(f"Degenerate MOS range in {name}")
        df = df.copy()
        df["mos"] = 1.0 + 4.0 * (df["mos"] - lo) / (hi - lo)
        extras.append(df)
    if not extras:
        return pd.DataFrame()
    return pd.concat(extras, ignore_index=True)


# ---------------------------------------------------------------------------
# Transforms and dataset
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(
    size: int,
    train: bool,
    norm: str = "none",
    augment: str = "standard",
    eval_mode: str = "center",
) -> T.Compose:
    """Build a quality-preserving transform.

    Aggressive crops/colour jitter are deliberately avoided: they change the
    perceived quality of the image, which would inject label noise.  Geometric
    augmentation is limited to horizontal flips and a one-sided crop.
    """
    ops: List[Any] = []
    if train:
        if augment == "standard":
            ops += [
                T.Resize(size + 32, antialias=True),
                T.RandomCrop(size),
                T.RandomHorizontalFlip(p=0.5),
            ]
        elif augment == "light":
            ops += [T.Resize(size + 16, antialias=True), T.RandomCrop(size)]
        elif augment == "none":
            ops += [T.Resize(size, antialias=True)]
        else:
            raise ValueError(f"Unknown augment mode: {augment}")
    else:
        if eval_mode == "center":
            ops += [T.Resize(size + 32, antialias=True), T.CenterCrop(size)]
        elif eval_mode == "resize":
            ops += [T.Resize((size, size), antialias=True)]
        elif eval_mode == "full":
            # Keep native resolution; all KADID/TID images share 512x384 so
            # batching is still possible.  Important for classical metrics
            # whose features depend on the full image (blockiness, ringing).
            pass
        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")

    ops.append(T.ToTensor())
    if norm == "imagenet":
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    elif norm != "none":
        raise ValueError(f"Unknown norm: {norm}")
    return T.Compose(ops)


class IQADataset(Dataset):
    """Torch dataset returning a dict per distorted image."""

    def __init__(self, df: pd.DataFrame, transform: Optional[Any] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as im:
            im = im.convert("RGB")
            img = self.transform(im) if self.transform is not None else T.ToTensor()(im)
        return {
            "image": img,
            "mos": float(row["mos"]),
            "ref": str(row["ref"]),
            "name": str(row["name"]),
            "dataset": str(row["dataset"]),
            "dist_type": int(row["dist_type"]),
            "level": int(row["level"]),
        }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def make_loader(
    df: pd.DataFrame,
    transform: Any,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = False,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Deterministic DataLoader wrapper."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = IQADataset(df, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def make_splits_and_loaders(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience constructor used by train/evaluate scripts.

    Returns a dict with the dataframes, datasets (for prediction bookkeeping)
    and the three loaders.  Evaluation loaders always use the clean protocol
    (center crop, no random augmentation).
    """
    data_cfg = cfg["data"]
    df = load_dataframe(data_cfg["dataset"], data_cfg["root"])
    train_df, val_df, test_df = split_dataframe(
        df,
        mode=data_cfg["split"],
        test_frac=data_cfg["test_frac"],
        val_frac=data_cfg["val_frac"],
        seed=cfg["experiment"]["seed"],
        fold=data_cfg["fold"],
        kfold=data_cfg["kfold"],
    )
    extra = load_extra_train(cfg, exclude=(data_cfg["dataset"],))
    if len(extra):
        train_df = pd.concat([train_df, extra], ignore_index=True)
        # Deterministic shuffle so dataset order does not bias batches
        train_df = train_df.sample(frac=1.0, random_state=int(cfg["experiment"]["seed"])).reset_index(drop=True)
        print(f"multi-dataset training: +{len(extra)} images from "
              f"{sorted(extra['dataset'].unique().tolist())}")
    size = int(data_cfg["size"])
    norm = data_cfg["norm"]
    train_tf = build_transform(
        size, train=True, norm=norm, augment=data_cfg["augment"]
    )
    eval_tf = build_transform(
        size, train=False, norm=norm, eval_mode=data_cfg["eval_mode"]
    )
    bs = int(cfg["train"]["batch_size"])
    nw = int(data_cfg["num_workers"])
    seed = int(cfg["experiment"]["seed"])
    loaders = {
        "train": make_loader(
            train_df, train_tf, bs, nw, shuffle=True, drop_last=cfg["train"]["drop_last"], seed=seed
        ),
        "val": make_loader(val_df, eval_tf, bs, nw, shuffle=False),
        "test": make_loader(test_df, eval_tf, bs, nw, shuffle=False),
    }
    return {
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "loaders": loaders,
        "eval_transform": eval_tf,
        "summary": split_summary(train_df, val_df, test_df),
    }
