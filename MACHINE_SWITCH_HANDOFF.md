# BrainSegNet Machine Switch Handoff

This document is your context pack for moving to a new computer and resuming quickly.

## Current Project State

- Main project: `BrainSegNet/dl_project_new`
- Baseline 1 (newly added): plain 3D U-Net in:
  - `dl_project_new/baselines/unet_baseline/model.py`
  - `dl_project_new/baselines/unet_baseline/train_baseline1.py`
  - `dl_project_new/baselines/unet_baseline/evaluate_baseline1.py`
- Baseline scripts were updated to support:
  - `--data-root` (so you do not depend on Docker `/app/...` paths)
  - Windows output fallback if config still points to Docker paths

## What You Must Copy to the New Machine

At minimum:

1. Entire folder: `BrainSegNet/`
2. BraTS dataset folder (NOT in git):
   - Must contain case folders like `BraTS20_Training_001`, `BraTS20_Training_002`, ...
   - Each case must have: `_t1`, `_t1ce`, `_t2`, `_flair`, `_seg` NIfTI files
3. Optional but recommended:
   - Existing outputs/checkpoints in `BrainSegNet/outputs/`

## Critical Reality Check (Before Training)

Training fails with `num_samples=0` if your data path is wrong.

The data-root value must point to the folder that directly contains the case folders.

Correct style:

`X:\...\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\...`

So your `--data-root` should be:

`X:\...\MICCAI_BraTS2020_TrainingData`

## New Machine Setup (PowerShell)

From your workspace root:

```powershell
cd .\BrainSegNet\dl_project_new
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

GPU sanity check:

```powershell
python -c "import torch; print(torch.__version__); print('cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_gpu')"
```

## Set Dataset Path (Do This Once Per Session)

```powershell
$DATA_ROOT="D:\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
```

Quick check path contents:

```powershell
Get-ChildItem "$DATA_ROOT" | Select-Object -First 5
```

You should see folders like `BraTS20_Training_001` (not just README files).

## Run Baseline 1 (U-Net)

Quick smoke run:

```powershell
python baselines/unet_baseline/train_baseline1.py --data-root "$DATA_ROOT" --epochs 3
```

Full training:

```powershell
python baselines/unet_baseline/train_baseline1.py --data-root "$DATA_ROOT"
```

Evaluate all 15 missing-modality combinations:

```powershell
python baselines/unet_baseline/evaluate_baseline1.py --data-root "$DATA_ROOT"
```

Baseline outputs:

- `outputs/baselines/unet_baseline/checkpoints/baseline1_best.pth`
- `outputs/baselines/unet_baseline/baseline1_history.json`
- `outputs/baselines/unet_baseline/baseline1_eval_results.json`

## Compare Baseline vs BrainSegNet

Run from `dl_project_new`:

```powershell
python -c "import json, pathlib; b=json.load(open(pathlib.Path('..')/'outputs'/'baselines'/'unet_baseline'/'baseline1_eval_results.json')); g=json.load(open(pathlib.Path('..')/'outputs'/'eval_results.json')); print('Baseline1 MEAN:', b['MEAN']); print('BrainSegNet MEAN:', g['MEAN']); print('Delta (BrainSegNet - Baseline1):', {k: round(g['MEAN'][k]-b['MEAN'][k],4) for k in ['WT','TC','ET']})"
```

## BrainSegNet Main Project Commands

Use only after data path is valid in `config.py` or by adapting scripts similarly:

```powershell
python train.py --mode teacher
python train.py --mode student
python evaluate.py
```

## Recommended Presentation Framing

- BrainSegNet is the main model for missing-modality segmentation.
- M3AE is the literature benchmark/reference problem.
- Report:
  - Baseline 1 (plain U-Net)
  - BrainSegNet
  - Same splits + same 15 missing-modality evaluation protocol
- Avoid claiming "beats M3AE" unless protocols are strictly matched.

## Fast Troubleshooting

### `ERROR: DATA_ROOT not found`
- Fix `--data-root` value.

### `num_samples=0`
- Dataset folder is empty or wrong level in path.

### `cuda_available False`
- Install CUDA PyTorch in active venv and verify NVIDIA driver.

### OOM error
- Reduce `CROP_SIZE` to 64 and/or `BATCH_SIZE` to 1 in `config.py`.

## Priority Order After Switching Machines

1. Confirm repo copied.
2. Confirm dataset copied and visible via `Get-ChildItem $DATA_ROOT`.
3. Create venv and install deps.
4. Run baseline smoke test (`--epochs 3`).
5. Run full baseline + evaluate.
6. Compare against BrainSegNet results.

