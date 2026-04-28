"""
config.py  —  Central Path Configuration for BrainSegNet
=========================================================
THIS IS THE ONLY FILE YOU NEED TO EDIT IF PATHS CHANGE.

Current setup (SJSU Docker Lab — Ashmitha's machine):
  Windows path : C:\\Users\\ashmitha\\Desktop\\ashmitha\\
  Docker mounts: C:\\Users\\ashmitha\\Desktop\\ashmitha  →  /app
  Inside Docker: everything is accessible under /app/

Data folder structure expected:
  /app/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
      BraTS20_Training_001/
          BraTS20_Training_001_t1.nii
          BraTS20_Training_001_t1ce.nii
          BraTS20_Training_001_t2.nii
          BraTS20_Training_001_flair.nii
          BraTS20_Training_001_seg.nii
      BraTS20_Training_002/
      ...

Docker run command used:
  docker run --gpus all -p 8888:8888
    -v "C:\\Users\\ashmitha\\Desktop\\ashmitha":/app
    <IMAGE_NAME>:<TAG>
"""

import os

# ============================================================
# PRIMARY PATHS — change these if the machine or data path changes
# ============================================================

# Root of the BraTS 2020 training data (inside Docker container)
DATA_ROOT = "C:/Users/arpana/Downloads/BrainSegNet/data/BraTS_PED_2024/converted_brats_style"

# Where checkpoints, logs, and results are saved (inside Docker container)
OUTPUT_DIR = "C:/Users/arpana/Downloads/BrainSegNet_backup_outputs/ped2024_finetune"

# ============================================================
# TRAINING HYPERPARAMETERS — safe defaults for the lab GPU
# ============================================================

CROP_SIZE         = 96      # 3D crop size. Use 64 if GPU OOM error appears.
BATCH_SIZE        = 2       # Reduce to 1 if GPU runs out of memory
CROPS_PER_PATIENT = 4       # Random crops per patient per epoch
MISSING_PROB      = 0.5     # Probability each modality is zeroed (student training)
NUM_WORKERS       = 4       # MUST be 0 inside Docker on Windows
BASE_FILTERS      = 32      # Encoder filter count at level 1
LATENT_DIM        = 128     # VAE latent dimension
TEACHER_EPOCHS    = 100     # Stage 1 epochs
STUDENT_EPOCHS    = 20      # Stage 2 epochs (backup fine-tune run)
SEED              = 42
# The GAN needs a much lower starting LR
STUDENT_LR      = 5e-5    # was 2e-4 — 4x lower
DISC_LR         = 1e-5    # was 1e-4 — 10x lower
# ============================================================
# 5-PATIENT TEST MODE — flip to True for quick verification
# ============================================================
TEST_MODE         = False   # Set True for 5-patient / 3-epoch test run
TEST_N_TRAIN      = 4       # Patients used for training in test mode
TEST_N_VAL        = 1       # Patients used for validation in test mode
TEST_EPOCHS       = 3       # Epochs in test mode
USE_GAN           = False
# ============================================================
# Derived paths — do not edit below this line
# ============================================================
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR        = os.path.join(OUTPUT_DIR, "logs")
TEACHER_CKPT   = os.path.join(CHECKPOINT_DIR, "teacher_best.pth")
STUDENT_CKPT   = os.path.join(CHECKPOINT_DIR, "student_best.pth")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)
