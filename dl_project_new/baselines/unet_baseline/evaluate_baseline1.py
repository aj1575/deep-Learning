import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import OUTPUT_DIR, CROP_SIZE, BASE_FILTERS
from dataset import get_dataloaders
from losses import dice_brats
from baselines.unet_baseline.model import UNetBaseline3D


def resolve_output_dir():
    if os.name == "nt" and OUTPUT_DIR.startswith("/app"):
        return os.path.abspath(os.path.join(ROOT, "..", "outputs"))
    return OUTPUT_DIR


ALL_COMBOS = [
    ("All 4 modalities", [1, 1, 1, 1]),
    ("Missing T1", [0, 1, 1, 1]),
    ("Missing T1ce", [1, 0, 1, 1]),
    ("Missing T2", [1, 1, 0, 1]),
    ("Missing FLAIR", [1, 1, 1, 0]),
    ("Missing T1+T1ce", [0, 0, 1, 1]),
    ("Missing T1+T2", [0, 1, 0, 1]),
    ("Missing T1+FLAIR", [0, 1, 1, 0]),
    ("Missing T1ce+T2", [1, 0, 0, 1]),
    ("Missing T1ce+FLAIR", [1, 0, 1, 0]),
    ("Missing T2+FLAIR", [1, 1, 0, 0]),
    ("Only T1", [1, 0, 0, 0]),
    ("Only T1ce", [0, 1, 0, 0]),
    ("Only T2", [0, 0, 1, 0]),
    ("Only FLAIR", [0, 0, 0, 1]),
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Local BraTS root path. Overrides config DATA_ROOT.",
    )
    return parser.parse_args()


def run_evaluation(model, loader, device):
    model.eval()
    results = {}
    for name, mask_vals in ALL_COMBOS:
        d = {"WT": [], "TC": [], "ET": []}
        print(f"  {name} mask={mask_vals}")
        for imgs, segs, _ in tqdm(loader, desc=name, leave=False):
            imgs = imgs.to(device)
            segs = segs.to(device)
            imgs_masked = imgs.clone()
            for ch, present in enumerate(mask_vals):
                if not present:
                    imgs_masked[:, ch] = 0.0

            with torch.no_grad():
                out = model(imgs_masked)
            m = dice_brats(out, segs)
            for key in d:
                d[key].append(m[key])
        avg = {k: float(np.mean(d[k])) for k in d}
        results[name] = avg
        print(f"    WT {avg['WT']:.4f} TC {avg['TC']:.4f} ET {avg['ET']:.4f}")

    results["MEAN"] = {
        key: float(np.mean([results[name][key] for name, _ in ALL_COMBOS]))
        for key in ["WT", "TC", "ET"]
    }
    return results


def main():
    args = get_args()
    output_dir = resolve_output_dir()
    ckpt = os.path.join(output_dir, "baselines", "unet_baseline", "checkpoints", "baseline1_best.pth")
    assert os.path.exists(ckpt), (
        f"Checkpoint not found: {ckpt}\n"
        "Train first: python baselines/unet_baseline/train_baseline1.py"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetBaseline3D(in_channels=4, n_classes=4, base_filters=BASE_FILTERS).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    print(f"Loaded baseline checkpoint: {ckpt}")
    print(f"Best WT in training: {state.get('best_wt', -1):.4f}")

    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        missing_prob=0.0,
        crop_size=CROP_SIZE,
    )
    results = run_evaluation(model, test_loader, device)

    out_path = os.path.join(output_dir, "baselines", "unet_baseline", "baseline1_eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation results -> {out_path}")
    print(
        "MEAN (all 15): "
        f"WT {results['MEAN']['WT']:.4f} "
        f"TC {results['MEAN']['TC']:.4f} "
        f"ET {results['MEAN']['ET']:.4f}"
    )


if __name__ == "__main__":
    main()
