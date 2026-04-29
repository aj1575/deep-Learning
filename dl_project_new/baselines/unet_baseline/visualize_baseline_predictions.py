import os
import sys
import argparse
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import OUTPUT_DIR, CROP_SIZE, BASE_FILTERS  # noqa: E402
from dataset import get_dataloaders  # noqa: E402
from baselines.unet_baseline.model import UNetBaseline3D  # noqa: E402


ALL_COMBOS: List[Tuple[str, List[int]]] = [
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


def resolve_output_dir() -> str:
    if os.name == "nt" and OUTPUT_DIR.startswith("/app"):
        return os.path.abspath(os.path.join(ROOT, "..", "outputs"))
    return OUTPUT_DIR


def get_args():
    p = argparse.ArgumentParser(description="Save Baseline1/Baseline2 prediction figures.")
    p.add_argument("--data-root", default=None, help="BraTS data root.")
    p.add_argument("--split-file", default=None, help="Optional split JSON.")
    p.add_argument(
        "--cases",
        nargs="+",
        default=["All 4 modalities", "Missing T1+T2", "Only FLAIR"],
        help="Subset of modality cases from evaluator naming.",
    )
    p.add_argument("--sample-index", type=int, default=0, help="Test sample index to visualize.")
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> UNetBaseline3D:
    model = UNetBaseline3D(in_channels=4, n_classes=4, base_filters=BASE_FILTERS).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def normalize_for_plot(x: np.ndarray) -> np.ndarray:
    vmin, vmax = np.percentile(x, [1, 99])
    if vmax <= vmin:
        return np.zeros_like(x)
    x = np.clip(x, vmin, vmax)
    return (x - vmin) / (vmax - vmin)


def main():
    args = get_args()
    out_root = resolve_output_dir()
    vis_dir = os.path.join(out_root, "baselines", "unet_baseline", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    combo_map: Dict[str, List[int]] = dict(ALL_COMBOS)
    for name in args.cases:
        if name not in combo_map:
            raise ValueError(f"Unknown case: {name}")

    b1_ckpt = os.path.join(out_root, "baselines", "unet_baseline", "checkpoints", "baseline1_best.pth")
    b2_ckpt = os.path.join(out_root, "baselines", "unet_baseline", "checkpoints", "baseline2_best.pth")
    if not os.path.exists(b1_ckpt) or not os.path.exists(b2_ckpt):
        raise FileNotFoundError("Missing baseline checkpoints. Train Baseline 1 and 2 first.")

    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        split_file=args.split_file,
        missing_prob=0.0,
        crop_size=CROP_SIZE,
    )
    sample = test_loader.dataset[args.sample_index]
    imgs, seg, _ = sample

    imgs = imgs.unsqueeze(0)
    seg = seg.numpy()
    z = seg.shape[-1] // 2
    channel_names = ["T1", "T1ce", "T2", "FLAIR"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b1 = load_model(b1_ckpt, device)
    b2 = load_model(b2_ckpt, device)

    for case_name in args.cases:
        mask_vals = combo_map[case_name]
        imgs_masked = imgs.clone()
        for ch, present in enumerate(mask_vals):
            if not present:
                imgs_masked[:, ch] = 0.0

        with torch.no_grad():
            pred1 = torch.argmax(b1(imgs_masked.to(device)), dim=1).squeeze(0).cpu().numpy()
            pred2 = torch.argmax(b2(imgs_masked.to(device)), dim=1).squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for ch in range(4):
            sl = imgs_masked[0, ch, :, :, z].numpy()
            axes[0, ch].imshow(normalize_for_plot(sl), cmap="gray")
            flag = "present" if mask_vals[ch] == 1 else "missing"
            axes[0, ch].set_title(f"{channel_names[ch]} ({flag})")
            axes[0, ch].axis("off")

        axes[1, 0].imshow(seg[:, :, z], cmap="nipy_spectral", vmin=0, vmax=3)
        axes[1, 0].set_title("Ground Truth")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(pred1[:, :, z], cmap="nipy_spectral", vmin=0, vmax=3)
        axes[1, 1].set_title("Baseline1 Pred")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(pred2[:, :, z], cmap="nipy_spectral", vmin=0, vmax=3)
        axes[1, 2].set_title("Baseline2 Pred")
        axes[1, 2].axis("off")

        overlay = normalize_for_plot(imgs_masked[0, 1, :, :, z].numpy())
        axes[1, 3].imshow(overlay, cmap="gray")
        axes[1, 3].imshow(pred2[:, :, z], cmap="nipy_spectral", vmin=0, vmax=3, alpha=0.35)
        axes[1, 3].set_title("Baseline2 Overlay")
        axes[1, 3].axis("off")

        fig.suptitle(f"Sample {args.sample_index} | {case_name} | mask={mask_vals}")
        fig.tight_layout()
        out_name = case_name.lower().replace("+", "_").replace(" ", "_")
        out_path = os.path.join(vis_dir, f"sample{args.sample_index}_{out_name}.png")
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
