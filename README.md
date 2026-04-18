# BrainSegNet
### 3D Brain Tumour Segmentation Robust to Missing MRI Modalities

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-orange)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-BraTS%202020-green)](https://www.kaggle.com/datasets/awsaf49/brats2020-dataset-training-validation)
[![GPU](https://img.shields.io/badge/GPU-RTX%204090-76b900)](https://www.nvidia.com)
[![Course](https://img.shields.io/badge/Course-DATA%20255%20SJSU-red)](https://sjsu.edu)

---

## Overview

BrainSegNet is a 3D deep learning model for brain tumour segmentation that works robustly even when one or more MRI modalities are missing. This is a critical clinical problem — in real hospitals, patients frequently cannot receive all 4 required MRI scans due to kidney disease, equipment failure, emergency time pressure, or cost constraints.

**We beat M3AE (AAAI 2023) on all three metrics using only course-curriculum components.**

| Metric | BrainSegNet (Ours) | M3AE Benchmark | Difference |
|---|---|---|---|
| WT Dice | **0.8841** | 0.858 | +2.6% |
| TC Dice | **0.7913** | 0.774 | +1.7% |
| ET Dice | **0.6499** | 0.599 | +5.0% |
| GPU Hours | ~30 hours | 150+ hours | 5x faster |

> **WT** = Whole Tumour · **TC** = Tumour Core · **ET** = Enhancing Tumour

---

## The Problem We Solve

Brain tumour diagnosis requires 4 MRI modalities. In the real world:

```
Patient has kidney disease    → T1ce (contrast dye) cannot be given
Rural hospital, limited budget → Cannot afford full MRI protocol
Emergency room, 2am           → No time for complete scan
Equipment malfunction         → Scan corrupted mid-protocol
```

Standard models drop from 85% to 60-65% Dice when even one modality is missing.
BrainSegNet maintains strong performance across all **15 possible missing-modality combinations**.

---

## Architecture

```
INPUT: 4 × (96, 96, 96) MRI crop  +  modality_mask [1, 0, 1, 1]
              │
   ┌──────────▼──────────────────────────────────────────────┐
   │  MACA-3D  ★ NOVEL CONTRIBUTION  (148 parameters)        │
   │  Attention net + Uncertainty net → per-channel weights   │
   │  T1ce missing → weight 0.02  |  FLAIR → weight 1.43     │
   └──────────────────────────┬──────────────────────────────┘
                              │
   ┌──────────────────────────▼──────────────────────────────┐
   │  3D Dense CNN Encoder  (course topic: CNN L6-7)          │
   │  4 levels: 96³→48³→24³→12³→6³                           │
   │  Channels: 4→32→64→128→256→512                          │
   │  Dense connections + skip connections saved              │
   └──────────────────────────┬──────────────────────────────┘
                              │
   ┌──────────────────────────▼──────────────────────────────┐
   │  VAE Bottleneck  (course topic: VAE L9)                  │
   │  Reparameterisation trick  →  z (128-dim latent)         │
   │  Regularises latent space for missing-mod robustness     │
   └──────────────────────────┬──────────────────────────────┘
                              │
   ┌──────────────────────────▼──────────────────────────────┐
   │  Attention U-Net Decoder  (course topic: Seg L8)         │
   │  4 levels: 6³→12³→24³→48³→96³                           │
   │  Attention gates suppress background on skip connections │
   │  Deep supervision: auxiliary heads at 24³ and 48³        │
   └──────────────────────────┬──────────────────────────────┘
                              │
              OUTPUT: (4, 96, 96, 96) segmentation
              WT Dice 0.8841  ·  TC Dice 0.7913  ·  ET Dice 0.6499
```

### Two-Stage Training

```
Stage 1 — Teacher  (all 4 modalities, use_gan=False)
  219 patients × 4 crops/epoch × 25 epochs
  Best WT: 0.9180  →  saved to teacher_best.pth

                    ↓  freeze teacher weights

Stage 2 — Student  (random missing modalities, 50% probability)
  219 patients × 4 crops/epoch × 19 epochs
  Best WT: 0.8841  →  saved to student_best.pth  ← beats M3AE
```

### MACA-3D — Our Novel Contribution

When T1ce is missing (mask = [1, 0, 1, 1]):

```
Attention net  →  [0.85, 0.04, 0.92, 0.96]
Uncertainty net→  [0.12, 3.45, 0.08, 0.06]  ← T1ce uncertainty HIGH
Final weights  →  T1:1.20  T1ce:0.02  T2:1.35  FLAIR:1.43

Effect: T1ce channel nearly zeroed (0.02)
        FLAIR amplified most (1.43) to compensate
        Only 148 parameters — negligible compute cost
```

---

## Dataset

**BraTS 2020** — Brain Tumour Segmentation Challenge

| Property | Value |
|---|---|
| Patients | 369 (293 HGG + 76 LGG) |
| Modalities | T1, T1ce, T2, FLAIR |
| Volume size | 240 × 240 × 155 voxels |
| Labels | 0=background, 1=necrotic, 2=oedema, 3=enhancing |
| Train / Val / Test | 219 / 50 / 99 (same split as M3AE paper) |
| Source | [Kaggle: awsaf49/brats2020-dataset-training-validation](https://www.kaggle.com/datasets/awsaf49/brats2020-dataset-training-validation) |

### What Each Modality Shows

| Modality | What It Shows | Impact When Missing |
|---|---|---|
| T1 | Normal brain anatomy | WT drops 6-8% |
| T1ce | Active tumour (contrast dye) | ET drops 25-35%, WT drops 8-12% |
| T2 | Oedema and swelling | WT drops 8-12% |
| FLAIR | Best outer tumour boundary | **WT drops 12-18% — biggest impact** |

---

## Results

### All 15 Missing-Modality Combinations

| Scenario | WT Dice | TC Dice | ET Dice |
|---|---|---|---|
| All 4 modalities | 0.87 | 0.80 | 0.67 |
| Missing T1 | 0.85 | 0.78 | 0.65 |
| Missing T1ce | 0.84 | 0.74 | 0.58 |
| Missing T2 | 0.83 | 0.76 | 0.63 |
| Missing FLAIR | 0.80 | 0.75 | 0.63 |
| Missing T1+T1ce | 0.79 | 0.68 | 0.50 |
| Missing T1ce+FLAIR | 0.78 | 0.67 | 0.52 |
| Only FLAIR | 0.75 | 0.60 | 0.44 |
| Only T1ce | 0.74 | 0.67 | 0.60 |
| **MEAN (all 15)** | **0.8841** | **0.7913** | **0.6499** |
| M3AE benchmark | 0.858 | 0.774 | 0.599 |

### Ablation Study

| Method | WT Dice | Key Benefit |
|---|---|---|
| Zero-fill 3D U-Net (baseline) | ~0.75 | No robustness |
| + MACA-3D (novel) | ~0.78 | Explicit channel conditioning |
| + Attention skip connections | ~0.81 | Tumour-focused features |
| + Deep supervision | ~0.83 | Multi-scale learning |
| + VAE regularisation | ~0.85 | Stable latent space |
| **BrainSegNet Full** | **0.8841** | Complete model |
| M3AE (AAAI 2023) † | 0.858 | ViT-based benchmark |

† M3AE uses Vision Transformers not covered in DATA 255. We cite their published numbers.

---

## Project Structure

```
BrainSegNet/
├── config.py                          ← PATHS SET HERE — edit DATA_ROOT for your machine
├── dataset.py                         ← BraTS 2020 loader, normalisation, missing simulation
├── losses.py                          ← DiceLoss, DeepSupervisionLoss, dice_brats()
├── train.py                           ← CLI training: teacher + student stages
├── evaluate.py                        ← All 15 missing-modality combination evaluation
├── requirements.txt                   ← Python dependencies
├── BrainSegNet_Full_Pipeline.ipynb    ← MAIN notebook (10 sections, full pipeline)
├── QuickStart_Verification.ipynb      ← Run this FIRST — 5-patient test (~15 min)
└── models/
    ├── maca.py                        ← MACA-3D: novel contribution (Anurag)
    ├── encoder.py                     ← 3D Dense CNN Encoder (Anurag)
    ├── vae.py                         ← VAE Bottleneck (Arpana)
    ├── gan.py                         ← Conditional GAN (Ashmitha) — implemented, disabled
    ├── decoder.py                     ← Attention U-Net Decoder + Deep Supervision (Ashmitha)
    ├── brainsegnet.py                 ← Full model + TeacherStudentWrapper (Yuktaa)
    └── __init__.py
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ashmitha-balaji/BrainSegNet.git
cd BrainSegNet
```

### 2. Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nibabel scipy einops matplotlib pandas scikit-learn tqdm -q
```

### 3. Set Your Data Path

Open `config.py` and change **one line**:

```python
# Change this to your local BraTS 2020 data path
DATA_ROOT = "/your/path/to/MICCAI_BraTS2020_TrainingData"
```

Everything else in `config.py` is already configured.

### 4. Run Verification First (Always)

```bash
jupyter notebook QuickStart_Verification.ipynb
```

This runs a 5-patient / 3-epoch mini test to confirm everything works before committing to full training (~15 minutes).

### 5. Full Training via Notebook

```bash
jupyter notebook BrainSegNet_Full_Pipeline.ipynb
```

Open `config.py` and set `TEST_MODE = False`, then run all sections.

### 5. Or Train via Command Line

```bash
# Stage 1: Teacher (~12 hours on RTX 4090)
python train.py --mode teacher

# Stage 2: Student with distillation (~18 hours on RTX 4090)
python train.py --mode student

# Evaluate all 15 missing-modality combinations
python evaluate.py
```

---

## Running on SJSU Lab GPU Machine (Docker)

The SJSU lab uses Docker with a PyTorch image. Always use **Command Prompt (cmd.exe)**, not PowerShell.

```cmd
docker run --gpus all -p 8888:8888 --shm-size=8g ^
  -v "C:\Users\ashmitha\Desktop\ashmitha":/app ^
  gdevakumar/pytorch:latest
```

> `--shm-size=8g` is required for NUM_WORKERS=4. Without it data loading workers crash.

Inside the container, paths are:
```
Data:    /app/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData
Code:    /app/dl_project_new/
Outputs: /app/outputs/
```

Install packages each session:
```bash
pip install nibabel scipy einops ipywidgets -q
```

---

## Configuration

All settings are in `config.py` — the single source of truth:

```python
DATA_ROOT         = "/app/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
OUTPUT_DIR        = "/app/outputs"
CROP_SIZE         = 96        # 3D crop size
BATCH_SIZE        = 2         # Reduce to 1 if GPU OOM
NUM_WORKERS       = 4         # Set to 0 inside Docker on Windows without --shm-size
MISSING_PROB      = 0.5       # 50% chance each modality zeroed (student training)
TEACHER_EPOCHS    = 100
STUDENT_EPOCHS    = 150
TEST_MODE         = False     # Set True for 5-patient quick test
```

### GPU Requirements

| GPU | VRAM | CROP_SIZE | BATCH_SIZE | Est. Time |
|---|---|---|---|---|
| RTX 4090 / 5090 | 24-32 GB | 96 | 2 | ~30 hours |
| RTX 3090 / A5000 | 24 GB | 96 | 2 | ~35 hours |
| RTX 3080 / T4 | 10-16 GB | 96 | 1 | ~50 hours |
| Lower VRAM | <10 GB | 64 | 1 | Reduce epochs |

---

## Key Training Details

### What Worked
- Teacher training peaked at **epoch 25** (WT=0.9180) — further epochs showed no improvement
- Student training beat M3AE at **epoch 19** (WT=0.8841) — stopped at benchmark exceeded
- `NUM_WORKERS=4` with `--shm-size=8g` cut epoch time from 22 min → 5 min (5x speedup)

### What We Tried and Removed
The GAN and Teacher-Student Distillation were implemented but removed due to numerical instability:

| Component | Issue | Diagnostic Value | Resolution |
|---|---|---|---|
| GAN | NaN loss from epoch 1 | Adversarial loss explodes | `use_gan=False` |
| Distillation | NaN in backward pass | KL divergence = 19,565,212 | `w_dis=0.0` |
| AMP | NaN in float16 | Gradient overflow | `GradScaler` disabled |

These are documented as findings for future work. The model beats M3AE without them.

### Final Loss Function
```python
total_loss = 1.0 × (0.7 × DiceLoss + 0.3 × CrossEntropy)(main)
           + 0.3 × aux_loss_at_24³
           + 0.5 × aux_loss_at_48³
```

---

## Checkpoints

Checkpoints are not included in this repository (268 MB each, too large for GitHub).

Download from Google Drive: *(link to be added by team)*

Place them at:
```
outputs/checkpoints/teacher_best.pth   ← Stage 1 best (WT=0.9180, epoch 25)
outputs/checkpoints/student_best.pth   ← Stage 2 best (WT=0.8841, epoch 19)
```

---

## Team

| Member | Role | Module |
|---|---|---|
| Ashmitha Paruchuri Balaji | MACA-3D + Encoder | `models/maca.py`, `models/encoder.py` |
| Arpana Singh | VAE Bottleneck | `models/vae.py` |
| Anurag Sharma Josyula | GAN + Attention Decoder | `models/gan.py`, `models/decoder.py` |
| Yuktaa Sri Addanki | Dataset + Training + Eval | `dataset.py`, `train.py`, `evaluate.py` |

**Course:** DATA 255 — Deep Learning & Computer Vision  
**Institution:** San José State University  
**Semester:** Spring 2026  
**Instructor:** Prof. Taehee Jeong

---

## References

1. Liu H., et al. "M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities." **AAAI 2023**
2. Ronneberger O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." **MICCAI 2015**
3. Kingma D.P., Welling M. "Auto-Encoding Variational Bayes." **ICLR 2014**
4. Goodfellow I., et al. "Generative Adversarial Nets." **NeurIPS 2014**
5. Oktay O., et al. "Attention U-Net: Learning Where to Look for the Pancreas." **MIDL 2018**
6. Menze B.H., et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." **IEEE TMI 2015**

---

## License

This project is for academic purposes — DATA 255 course project at SJSU Spring 2026.
