from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MoabbBCIIV2aConfig:
    # NOTE: MOABB paradigms interpret tmin/tmax relative to the dataset "task interval"
    # (BaseParadigm adds dataset.interval[0]).
    # For BNCI2014_001, dataset.interval == [2, 6] (seconds). Therefore:
    #   - To crop the canonical MI window 2–6s, use tmin=0, tmax=4 here.
    tmin: float = 0.0
    tmax: float = 4.0
    fmin: float = 4.0
    fmax: float = 38.0
    resample: int | None = None
    apply_car: bool = True
    cache_dir: str = "outputs/cache"


def _cache_key(cfg: MoabbBCIIV2aConfig) -> str:
    rs = "none" if cfg.resample is None else str(cfg.resample)
    return f"bnci2014_001_t{cfg.tmin:.3f}-{cfg.tmax:.3f}_f{cfg.fmin:.3f}-{cfg.fmax:.3f}_r{rs}_car{int(cfg.apply_car)}"


def load_bnci2014_001(cfg: MoabbBCIIV2aConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        x: (N, C, T)
        y: (N,) int in [0,3]
        subject: (N,) int in [1..9]
    """
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(cfg)
    npz_path = cache_dir / f"{key}.npz"
    meta_path = cache_dir / f"{key}.meta.json"

    if npz_path.exists():
        data = np.load(npz_path)
        x = data["x"]
        y = data["y"]
        subject = data["subject"]
        return x, y, subject

    # Import moabb lazily to keep module import light.
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        resample=cfg.resample,
    )

    subjects = list(dataset.subject_list)
    x, y_raw, meta = paradigm.get_data(dataset=dataset, subjects=subjects)

    # y might be strings; map to int labels with paradigm.classes order for determinism.
    classes = list(getattr(paradigm, "classes", [])) or sorted(set(y_raw))
    class_to_int = {c: i for i, c in enumerate(classes)}
    y = np.asarray([class_to_int[v] for v in y_raw], dtype=np.int64)

    if "subject" not in meta.columns:
        raise RuntimeError(f"MOABB meta is missing 'subject' column: {list(meta.columns)}")
    subject = meta["subject"].to_numpy(dtype=np.int64, copy=True)

    if cfg.apply_car:
        # Common average reference: subtract mean across channels at each time sample.
        x = x - x.mean(axis=1, keepdims=True)

    x = x.astype(np.float32, copy=False)

    np.savez_compressed(npz_path, x=x, y=y, subject=subject)
    meta = {
        "classes": classes,
        "n_samples": int(x.shape[0]),
        "n_channels": int(x.shape[1]),
        "n_times": int(x.shape[2]),
        "cfg": cfg.__dict__,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return x, y, subject


def get_loso_split_indices(
    subject: np.ndarray,
    *,
    target_subject: int,
    source_val_subject: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Strict LOSO (outer): target_subject is test-only.
    Inner: pick one source subject as val (source_val_subject).
    """
    if subject.ndim != 1:
        raise ValueError("subject must be 1D")

    all_subjects = np.unique(subject).tolist()
    if target_subject not in all_subjects:
        raise ValueError(f"target_subject={target_subject} not found in subjects={all_subjects}")

    test_mask = subject == target_subject
    source_mask = ~test_mask
    source_subjects = np.unique(subject[source_mask]).tolist()

    if source_val_subject is None:
        source_val_subject = int(source_subjects[0])
    if source_val_subject not in source_subjects:
        raise ValueError(f"source_val_subject={source_val_subject} not in source_subjects={source_subjects}")

    val_mask = subject == source_val_subject
    train_mask = source_mask & (~val_mask)

    return {
        "train": np.flatnonzero(train_mask),
        "val": np.flatnonzero(val_mask),
        "test": np.flatnonzero(test_mask),
    }
