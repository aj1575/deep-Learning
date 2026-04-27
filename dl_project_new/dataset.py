"""
dataset.py  —  BraTS 2020 Dataset Loader
=========================================
Author : Yuktaa Sri Addanki
Course : DATA 255 Deep Learning, SJSU Spring 2026

PATHS CHANGED FROM ORIGINAL:
  - All paths are now read from config.py (DATA_ROOT, NUM_WORKERS)
  - No hardcoded paths in this file
  - config.py is the single source of truth for DATA_ROOT

Docker path mapping:
  Windows:  C:\\Users\\ashmitha\\Desktop\\ashmitha\\data\\BraTS2020_TrainingData\\...
  Docker:   /app/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData
  (set in config.py as DATA_ROOT)
"""

import os
import random
import json
from typing import List, Tuple

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

# Import paths from central config
from config import (DATA_ROOT, OUTPUT_DIR, CROP_SIZE, BATCH_SIZE,
                    CROPS_PER_PATIENT, MISSING_PROB, NUM_WORKERS, SEED,
                    TEST_MODE, TEST_N_TRAIN, TEST_N_VAL)


# ------------------------------------------------------------------
# File loading helpers
# ------------------------------------------------------------------

def _load_nii(path_stem: str) -> np.ndarray:
    """Try .nii then .nii.gz — BraTS 2020 Kaggle version uses .nii"""
    for ext in ('.nii', '.nii.gz'):
        p = path_stem + ext
        if os.path.exists(p):
            return nib.load(p).get_fdata(dtype=np.float32)
    raise FileNotFoundError(
        f"Cannot find {path_stem}.nii or {path_stem}.nii.gz\n"
        f"Check DATA_ROOT in config.py: {DATA_ROOT}"
    )


def load_patient(patient_dir: str, pid: str
                 ) -> Tuple[np.ndarray, np.ndarray]:
    base  = os.path.join(patient_dir, pid)
    t1    = _load_nii(base + '_t1')
    t1ce  = _load_nii(base + '_t1ce')
    t2    = _load_nii(base + '_t2')
    flair = _load_nii(base + '_flair')
    seg   = _load_nii(base + '_seg')
    return np.stack([t1, t1ce, t2, flair], axis=0), seg


def z_score(vol: np.ndarray) -> np.ndarray:
    mask = vol > 0
    if not mask.any():
        return vol
    m, s = vol[mask].mean(), vol[mask].std()
    if s < 1e-8:
        return vol
    out       = np.zeros_like(vol)
    out[mask] = (vol[mask] - m) / s
    return out


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """BraTS uses {0,1,2,4}. Remap 4→3 for PyTorch cross-entropy."""
    out       = seg.copy()
    out[seg == 4] = 3
    return out.astype(np.int64)


def tumour_centre(seg: np.ndarray) -> List[int]:
    coords = np.where(seg > 0)
    if len(coords[0]) == 0:
        return [s // 2 for s in seg.shape]
    return [int(np.mean(c)) for c in coords]


def random_crop(image: np.ndarray, seg: np.ndarray,
                crop: int = 96, bias: float = 0.80):
    H, W, D = image.shape[1:]
    c, jit  = crop, crop // 4

    if random.random() < bias:
        cx, cy, cz = tumour_centre(seg)
        cx += random.randint(-jit, jit)
        cy += random.randint(-jit, jit)
        cz += random.randint(-jit, jit)
        x0 = max(0, min(cx - c // 2, H - c))
        y0 = max(0, min(cy - c // 2, W - c))
        z0 = max(0, min(cz - c // 2, D - c))
    else:
        x0 = random.randint(0, max(0, H - c))
        y0 = random.randint(0, max(0, W - c))
        z0 = random.randint(0, max(0, D - c))

    return (image[:, x0:x0+c, y0:y0+c, z0:z0+c],
            seg[x0:x0+c, y0:y0+c, z0:z0+c],
            (x0, y0, z0))


# ------------------------------------------------------------------
# Patient discovery and split
# ------------------------------------------------------------------

def find_valid_patients(data_root: str = None) -> List[str]:
    """Scan data_root and return list of valid patient IDs."""
    root  = data_root or DATA_ROOT
    valid = []
    if not os.path.isdir(root):
        print(f"ERROR: DATA_ROOT not found: {root}")
        print("Check config.py — DATA_ROOT must match your Docker mount path.")
        return valid
    for pid in sorted(os.listdir(root)):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue
        ok = all(
            any(os.path.exists(os.path.join(pdir, f'{pid}_{m}{e}'))
                for e in ('.nii', '.nii.gz'))
            for m in ['t1', 't1ce', 't2', 'flair', 'seg']
        )
        if ok:
            valid.append(pid)
    return valid


def get_splits(data_root: str = None, seed: int = SEED):
    """Standard 219/50/100 split matching M3AE paper."""
    all_p = find_valid_patients(data_root or DATA_ROOT)
    random.seed(seed)
    random.shuffle(all_p)
    return all_p[:219], all_p[219:269], all_p[269:]


def load_splits_from_file(split_file: str) -> Tuple[List[str], List[str], List[str]]:
    """Load fixed patient splits from JSON file with train/val/test keys."""
    with open(split_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    missing = [k for k in ("train", "val", "test") if k not in payload]
    if missing:
        raise ValueError(f"Split file missing keys: {missing}")
    return payload["train"], payload["val"], payload["test"]


# ------------------------------------------------------------------
# PyTorch Dataset
# ------------------------------------------------------------------

class BraTS2020Dataset(Dataset):
    """
    BraTS 2020 3-D dataset with missing-modality simulation.

    num_workers MUST be 0 when running inside Docker on Windows.
    This is set automatically from config.py (NUM_WORKERS = 0).
    """

    def __init__(self, patient_ids, data_root=None, crop_size=None,
                 crops_per_patient=None, split='train', missing_prob=None):
        self.ids   = patient_ids
        self.root  = data_root         or DATA_ROOT
        self.crop  = crop_size         or CROP_SIZE
        self.cpp   = crops_per_patient or CROPS_PER_PATIENT
        self.split = split
        self.mp    = (missing_prob if missing_prob is not None
                      else MISSING_PROB) if split == 'train' else 0.0

        print(f"[{split}] {len(patient_ids)} patients "
              f"× {self.cpp} crops = {len(self)} samples  "
              f"| DATA_ROOT: {self.root}")

    def __len__(self):
        return len(self.ids) * self.cpp

    def __getitem__(self, idx):
        pid  = self.ids[idx % len(self.ids)]
        pdir = os.path.join(self.root, pid)
        img, seg = load_patient(pdir, pid)
        img  = np.stack([z_score(img[i]) for i in range(4)], 0)
        seg  = remap_labels(seg)
        img, seg, _ = random_crop(img, seg, self.crop)

        mask = np.ones(4, dtype=np.float32)
        if self.mp > 0:
            while True:
                drop = np.random.rand(4) < self.mp
                if not drop.all():
                    break
            img[drop]  = 0.0
            mask[drop] = 0.0

        return (torch.from_numpy(img.astype(np.float32)),
                torch.from_numpy(seg.astype(np.int64)),
                torch.from_numpy(mask))


# ------------------------------------------------------------------
# DataLoader factory — reads all settings from config.py
# ------------------------------------------------------------------

def get_dataloaders(data_root=None, batch_size=None, num_workers=None,
                    crop_size=None, crops_per_patient=None,
                    seed=None, missing_prob=None, split_file=None):
    """
    Create train/val/test DataLoaders.

    All parameters default to values in config.py.
    num_workers is forced to 0 inside Docker on Windows.
    """
    root    = data_root         or DATA_ROOT
    bs      = batch_size        or BATCH_SIZE
    nw      = num_workers       if num_workers is not None else NUM_WORKERS
    cs      = crop_size         or CROP_SIZE
    cpp     = crops_per_patient or CROPS_PER_PATIENT
    sd      = seed              or SEED
    mp      = missing_prob      if missing_prob is not None else MISSING_PROB

    if TEST_MODE:
        print("TEST_MODE=True — using 5-patient mini dataset")
        all_p   = find_valid_patients(root)
        tr_ids  = all_p[:TEST_N_TRAIN]
        va_ids  = all_p[TEST_N_TRAIN:TEST_N_TRAIN + TEST_N_VAL]
        te_ids  = va_ids
        cpp_tr  = 2
    else:
        if split_file:
            tr_ids, va_ids, te_ids = load_splits_from_file(split_file)
        else:
            tr_ids, va_ids, te_ids = get_splits(root, sd)
        cpp_tr  = cpp

    ds_tr = BraTS2020Dataset(tr_ids, root, cs, cpp_tr, 'train', mp)
    ds_va = BraTS2020Dataset(va_ids, root, cs, 1,      'val',   0.0)
    ds_te = BraTS2020Dataset(te_ids, root, cs, 1,      'test',  0.0)

    kw = dict(num_workers=nw, pin_memory=(nw > 0))
    return (DataLoader(ds_tr, bs,  shuffle=True,  **kw),
            DataLoader(ds_va, 1,   shuffle=False, **kw),
            DataLoader(ds_te, 1,   shuffle=False, **kw))
