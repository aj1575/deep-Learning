# Baseline 1: Plain 3D U-Net

This baseline is a standard encoder-decoder 3D U-Net trained on the same BraTS data pipeline as BrainSegNet.

## Fair comparison choices

- Same dataset loader: `dataset.get_dataloaders()`
- Same split logic: train/val/test from `dataset.py`
- Same segmentation metrics: WT/TC/ET Dice from `losses.py`
- Missing-modality robustness tested on the same 15 combinations

Baseline-1 training uses full modalities only (`missing_prob=0.0`).
At evaluation time, missing modalities are simulated by zero-filling channels.

## Run (Windows PowerShell)

From `dl_project_new`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Train baseline:

```powershell
python baselines/unet_baseline/train_baseline1.py --data-root "D:\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
```

Evaluate on all 15 missing-modality scenarios:

```powershell
python baselines/unet_baseline/evaluate_baseline1.py --data-root "D:\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
```

## Outputs

- Checkpoint: `outputs/baselines/unet_baseline/checkpoints/baseline1_best.pth`
- Training history: `outputs/baselines/unet_baseline/baseline1_history.json`
- Evaluation table json: `outputs/baselines/unet_baseline/baseline1_eval_results.json`
