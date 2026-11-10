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




def _koniq_rows(root: str) -> List[Dict[str, Any]]:
    """KonIQ-10k authentic-distortion labels (DaRUS 72288) + 512x384 images."""
    label_path = os.path.join(root, "koniq10k_scores_and_distributions.tab")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"KonIQ label file not found: {label_path}")
    image_root = os.path.join(root, "koniq10k_512x384")
    if not os.path.isdir(image_root):
        image_root = root
    index: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(image_root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                index.setdefault(f.lower(), os.path.join(dirpath, f))
    if not index:
        raise FileNotFoundError(f"No KonIQ images found under {image_root}")
    rows: List[Dict[str, Any]] = []
    with open(label_path, "r", encoding="utf-8", errors="ignore") as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            name = parts[0].strip().strip('"')
            key = name.lower()
            if key not in index:
                continue
            rows.append(
                {
                    "dataset": "koniq10k",
                    "name": name,
                    "path": index[key],
                    "ref": name,
                    "mos": float(parts[7]),
                    "mos_std": float(parts[8]) if len(parts) > 8 and parts[8] else np.nan,
                    "dist_type": 0,
                    "level": 0,
                }
            )
    if not rows:
        raise ValueError(f"No KonIQ rows matched under {root}")
    return rows




def _livec_rows(root: str) -> List[Dict[str, Any]]:
    """LIVE Challenge (LIVEC) authentic-distortion dataset.

    Expects the IQA-Toolbox layout: ``Data/AllImages_release.mat``,
    ``Data/AllMOS_release.mat`` and an image folder (``Images`` or ``images``).
    """
    import scipy.io as sio

    data_dir = os.path.join(root, "Data")
    if not os.path.isdir(data_dir):
        data_dir = root
    names = sio.loadmat(os.path.join(data_dir, "AllImages_release.mat"))["AllImages_release"]
    mos = sio.loadmat(os.path.join(data_dir, "AllMOS_release.mat"))["AllMOS_release"][0]
    std = sio.loadmat(os.path.join(data_dir, "AllStdDev_release.mat"))["AllStdDev_release"][0]
    img_dirs = [d for d in (os.path.join(root, "Images"), os.path.join(root, "images"),
                            os.path.join(root, "Data", "Images"), root) if os.path.isdir(d)]
    index: Dict[str, str] = {}
    for d in img_dirs:
        for dirpath, _, filenames in os.walk(d):
            for f in filenames:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    index.setdefault(f.lower(), os.path.join(dirpath, f))
        if index:
            break
    if not index:
        raise FileNotFoundError(f"No LIVEC images found under {root}")
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(names):
        name = str(item[0][0])
        key = name.lower()
        if key not in index:
            continue
        rows.append(
            {
                "dataset": "livec",
                "name": name,
                "path": index[key],
                "ref": name,
                "mos": float(mos[i]),
                "mos_std": float(std[i]),
                "dist_type": 0,
                "level": 0,
            }
        )
    if not rows:
        raise ValueError(f"No LIVEC rows matched under {root}")
    return rows


def _spaq_rows(root: str) -> List[Dict[str, Any]]:
    """SPAQ smartphone-photography dataset (MOS 1-100)."""
    label_candidates = [
        os.path.join(root, "Annotations", "MOS and Image attribute scores.xlsx"),
        os.path.join(root, "MOS and Image attribute scores.xlsx"),
        os.path.join(root, "Annotations", "MOS_and_Image_attribute_scores.xlsx"),
    ]
    label_path = next((p for p in label_candidates if os.path.exists(p)), None)
    if label_path is None:
        # fall back to a CSV next to the images
        csvs = [p for p in (
            os.path.join(root, "annotations.csv"),
            os.path.join(root, "spaq_labels.csv"),
        ) if os.path.exists(p)]
        if not csvs:
            raise FileNotFoundError(f"SPAQ annotations not found under {root}")
        labels = pd.read_csv(csvs[0])
    else:
        labels = pd.read_excel(label_path)
    cols = {str(c).strip().lower(): c for c in labels.columns}
    name_col = next((cols[k] for k in ("image", "image_name", "img", "img_name", "file", "filename") if k in cols), labels.columns[0])
    mos_col = next((cols[k] for k in ("mos", "quality", "mos_quality") if k in cols), None)
    if mos_col is None:
        raise ValueError(f"SPAQ label file has no MOS column: {list(labels.columns)}")
    img_root = os.path.join(root, "TestImage")
    if not os.path.isdir(img_root):
        img_root = root
    index: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(img_root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                index.setdefault(f.lower(), os.path.join(dirpath, f))
    if not index:
        raise FileNotFoundError(f"No SPAQ images found under {img_root}")
    rows: List[Dict[str, Any]] = []
    for name_val, mos_val in zip(labels[name_col].tolist(), labels[mos_col].tolist()):
        name = str(name_val).strip()
        key = os.path.basename(name).lower()
        if key not in index:
            # some annotations use a bare numeric id
            key2 = (key + ".jpg") if not key.endswith((".jpg", ".jpeg", ".png")) else key
            if key2 not in index:
                continue
            key = key2
        rows.append(
            {
                "dataset": "spaq",
                "name": os.path.basename(index[key]),
                "path": index[key],
                "ref": os.path.basename(index[key]),
                "mos": float(mos_val),
                "mos_std": float("nan"),
                "dist_type": 0,
                "level": 0,
            }
        )
    if not rows:
        raise ValueError(f"No SPAQ rows matched under {root}")
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
    elif dataset in ("koniq10k", "koniq", "koniq-10k"):
        rows = _koniq_rows(root)
    elif dataset in ("livec", "live-challenge", "livechallenge"):
        rows = _livec_rows(root)
    elif dataset in ("spaq", "spaq10k"):
        rows = _spaq_rows(root)
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
        elif name.lower() in ("koniq10k", "koniq"):
            df = load_dataframe("koniq10k", data_cfg.get("koniq_root", "data/koniq"))
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
        elif eval_mode == "multi":
            # Larger resize without cropping; the evaluator takes several
            # fixed-size crops and averages their predictions.
            ops += [T.Resize(size + 64, antialias=True)]
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

    def __init__(self, df: pd.DataFrame, transform: Optional[Any] = None,
                 feature_npz: Optional[str] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.features = None
        self.feature_index = {}
        if feature_npz:
            if not os.path.exists(feature_npz):
                raise FileNotFoundError(f"Teacher feature file not found: {feature_npz}")
            with np.load(feature_npz) as z:
                names = [str(n) for n in z["names"]]
                self.features = z["features"].astype(np.float32)
            self.feature_index = {n: i for i, n in enumerate(names)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as im:
            im = im.convert("RGB")
            img = self.transform(im) if self.transform is not None else T.ToTensor()(im)
        item = {
            "image": img,
            "mos": float(row["mos"]),
            "ref": str(row["ref"]),
            "name": str(row["name"]),
            "dataset": str(row["dataset"]),
            "dist_type": int(row["dist_type"]),
            "level": int(row["level"]),
        }
        if "soft" in row:
            item["soft"] = float(row["soft"]) if pd.notna(row["soft"]) else float("nan")
        if self.features is not None:
            idx = self.feature_index.get(str(row["name"]))
            if idx is None:
                # Extra datasets (KonIQ) have no teacher feature; use zeros and
                # a validity flag so the loss can mask them out.
                item["tfeat"] = torch.zeros(self.features.shape[1], dtype=torch.float32)
                item["has_tfeat"] = 0.0
            else:
                item["tfeat"] = torch.from_numpy(self.features[idx])
                item["has_tfeat"] = 1.0
        return item


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
    feature_npz: Optional[str] = None,
) -> DataLoader:
    """Deterministic DataLoader wrapper."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = IQADataset(df, transform=transform, feature_npz=feature_npz)
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




class GroupedBatchSampler(torch.utils.data.Sampler):
    """Sample batches with several distorted versions of the same content.

    Each epoch cycles through every reference several times, drawing
    ``per_ref`` images per content and ``refs_per_batch`` contents per batch.
    This covers the whole dataset (with a small amount of overlap at content
    boundaries) while guaranteeing that every batch contains multiple
    distortions of the same reference for the content-grouped ordinal loss.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        refs_per_batch: int = 8,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.batch_size = int(batch_size)
        self.refs_per_batch = max(1, int(refs_per_batch))
        self.per_ref = max(1, self.batch_size // self.refs_per_batch)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.groups: Dict[str, List[int]] = {}
        for i, ref in enumerate(df["ref"].astype(str).tolist()):
            self.groups.setdefault(ref, []).append(i)
        self.refs = list(self.groups)
        self._epoch = 0
        max_len = max(len(v) for v in self.groups.values())
        self.passes = int(np.ceil(max_len / self.per_ref))
        if self.drop_last:
            per_pass = len(self.refs) // self.refs_per_batch
        else:
            per_pass = int(np.ceil(len(self.refs) / self.refs_per_batch))
        self._len = self.passes * max(1, per_pass)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        pools = {ref: rng.permutation(idx).tolist() for ref, idx in self.groups.items()}
        ptr = {ref: 0 for ref in self.refs}
        for _ in range(self.passes):
            order = rng.permutation(len(self.refs))
            for start in range(0, len(order), self.refs_per_batch):
                batch: List[int] = []
                for ridx in order[start:start + self.refs_per_batch]:
                    ref = self.refs[ridx]
                    pool = pools[ref]
                    if ptr[ref] + self.per_ref > len(pool):
                        pool = rng.permutation(self.groups[ref]).tolist()
                        pools[ref] = pool
                        ptr[ref] = 0
                    take = min(self.per_ref, len(pool) - ptr[ref])
                    batch.extend(pool[ptr[ref]:ptr[ref] + take])
                    ptr[ref] += take
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                rng.shuffle(batch)
                yield batch


def make_grouped_loader(
    df: pd.DataFrame,
    transform: Any,
    batch_size: int,
    refs_per_batch: int = 8,
    num_workers: int = 8,
    drop_last: bool = True,
    seed: int = 42,
    feature_npz: Optional[str] = None,
) -> DataLoader:
    dataset = IQADataset(df, transform=transform, feature_npz=feature_npz)
    sampler = GroupedBatchSampler(
        df, batch_size=batch_size, refs_per_batch=refs_per_batch,
        seed=seed, drop_last=drop_last,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
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
    soft_csv = data_cfg.get("soft_csv")
    if soft_csv:
        soft_path = str(soft_csv).format(fold=int(data_cfg.get("fold", 0)))
        if not os.path.exists(soft_path):
            raise FileNotFoundError(f"soft-target file not found: {soft_path}")
        soft = pd.read_csv(soft_path)
        if "name" not in soft or "soft" not in soft:
            raise ValueError(f"{soft_path} must contain name and soft columns")
        train_df = train_df.merge(soft[["name", "soft"]], on="name", how="left")
        missing = int(train_df["soft"].isna().sum())
        print(f"soft targets: {len(train_df) - missing}/{len(train_df)} images from {soft_path}"
              + (f" ({missing} extra images use hard labels only)" if missing else ""))
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
    feature_npz = data_cfg.get("teacher_features")
    if feature_npz:
        feature_npz = str(feature_npz).format(fold=int(data_cfg.get("fold", 0)))
        print(f"teacher features: {feature_npz}")
    if bool(data_cfg.get("grouped_batch", False)):
        train_loader = make_grouped_loader(
            train_df, train_tf, bs,
            refs_per_batch=int(data_cfg.get("refs_per_batch", 8)),
            num_workers=nw, drop_last=cfg["train"]["drop_last"], seed=seed,
            feature_npz=feature_npz,
        )
    else:
        train_loader = make_loader(
            train_df, train_tf, bs, nw, shuffle=True,
            drop_last=cfg["train"]["drop_last"], seed=seed,
            feature_npz=feature_npz,
        )
    loaders = {
        "train": train_loader,
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
