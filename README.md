# BrainSegNet End-to-End Project Report

This repository contains the full implementation and evaluation workflow for BrainSegNet and strong U-Net baselines under missing-modality MRI settings, including cross-domain testing on BraTS 2020 and BraTS PED 2024.

## 1) What We Built

- **Main model (`BrainSegNet`)**: MACA + 3D encoder + VAE bottleneck + decoder (`dl_project_new/models/brainsegnet.py`).
- **Baseline 1**: plain 3D U-Net trained with all modalities (`dl_project_new/baselines/unet_baseline/train_baseline1.py`).
- **Baseline 2**: same U-Net but trained with modality dropout (`dl_project_new/baselines/unet_baseline/train_baseline2.py`).
- **Unified evaluator** for all models and all modality-missing combinations: `dl_project_new/tools/evaluate_all_models.py`.
- **Fair-comparison utilities**: locked split generation, protocol configs, M3AE adapter, and PED-2024 conversion tools.

## 2) Why Baselines and M3AE

- **Baseline 1 purpose**: minimum reference for segmentation quality without robustness tricks.
- **Baseline 2 purpose**: test if simple modality dropout training improves robustness.
- **M3AE purpose**: strong external benchmark for missing-modality segmentation; used to establish fair method-level comparison rules (same modalities, same splits, same preprocessing, same metrics).

## 3) BrainSegNet I/O and Data Flow

### Input to model

- 4-channel 3D MRI crop: `[T1, T1ce, T2, FLAIR]` with shape `[B, 4, 96, 96, 96]`.
- Modality mask: `[B, 4]`, where `1=present`, `0=missing`.

### Output from model

- 4-class voxel logits `[B, 4, 96, 96, 96]`:
  - class 0: background
  - class 1/2/3: BraTS tumor regions (with label remap `4 -> 3` in preprocessing).
- During training mode, model returns auxiliary outputs (deep supervision + VAE terms).

### Data preparation and loader behavior

- NIfTI loading (`.nii`/`.nii.gz`) via `dataset.py`.
- Per-modality z-score normalization over non-zero voxels.
- Label remap from BraTS format `{0,1,2,4}` to `{0,1,2,3}`.
- Tumor-biased random crop (default `96^3`).
- Missing-modality simulation by zeroing channels according to mask.

## 4) Training and Testing Data

- **BraTS 2020** used as primary train/validation/test domain:
  - fixed split: train/val/test = `219/50/99` in `outputs/fair_splits_brats2020.json`.
- **BraTS PED 2024** used as cross-domain generalization test:
  - converted to BraTS-style layout with `dl_project_new/tools/convert_brats_ped2024_to_brats2020_layout.py`.
  - fixed split: `200/30/30` in `outputs/fair_splits_brats_ped2024.json`.

## 5) What We Achieved (Baselines vs BrainSegNet)

Using the saved reports:
- `outputs/all_models_eval_results_brats2020.json`
- `outputs/all_models_eval_results_brats_ped2024.json`

Mean over 15 modality settings:

- **Baseline1**
  - BraTS2020: WT `0.567`, TC `0.397`, ET `0.391`
  - PED2024: WT `0.210`, TC `0.169`, ET `0.229`
- **Baseline2**
  - BraTS2020: WT `0.706`, TC `0.511`, ET `0.374`
  - PED2024: WT `0.155`, TC `0.063`, ET `0.035`
- **BrainSegNet**
  - BraTS2020: WT `0.822`, TC `0.714`, ET `0.603`
  - PED2024: WT `0.208`, TC `0.072`, ET `0.017`

Interpretation:
- On in-domain BraTS2020, BrainSegNet is strongest overall.
- On PED2024, all models degrade heavily (domain shift stress-test).
- This confirms robustness work is useful but domain adaptation is still needed.

## 6) Reproducible Comparison Commands

Run full model comparison on a dataset:

```bash
python dl_project_new/tools/evaluate_all_models.py --data-root <DATA_ROOT> --brainsegnet-ckpt <PATH_TO_STUDENT_PTH>
```

Generate side-by-side 2020 vs 2024 tables:

```bash
python dl_project_new/tools/report_brats2020_vs_ped2024.py
```

## 7) Are We Done?

For this project phase, **yes**:
- baselines built,
- main model evaluated,
- cross-domain test done,
- fair comparison scaffolding added,
- reproducible report scripts provided.

For research/production, **not yet**:
- needs external validation on additional sites/scanners,
- calibration and uncertainty reporting,
- post-processing and clinical reporting layer.

## 8) Expected Questions and Answers

### Is BrainSegNet novel?

The model is a hybrid combination with explicit missing-modality conditioning (MACA), VAE regularization, and segmentation decoder. Novelty claim should be framed as **project-level architectural integration and evaluation protocol**, not as a fully new foundational architecture class.

### Can model outputs be understood by non-doctors?

Outputs are segmentation masks/probabilities, not final diagnosis text. They are useful visual aids but must be interpreted with clinical context.

### What exactly is evaluated?

Dice for BraTS subregions:
- WT (whole tumor),
- TC (tumor core),
- ET (enhancing tumor).
Evaluation runs across 15 modality availability patterns.

### Why test on BraTS PED 2024 if trained on BraTS 2020?

To quantify domain shift and generalization limits. It is a stress-test, not a like-for-like leaderboard claim.

## 9) Future Work

- Domain adaptation from adult glioma (BraTS2020) to pediatric/other domains.
- Robust calibration and uncertainty-aware decision support.
- Stronger harmonization and scanner normalization.
- Fair M3AE re-training and full side-by-side results under one locked protocol.
- Convert segmentation output into clinician-facing structured reports and overlays.
