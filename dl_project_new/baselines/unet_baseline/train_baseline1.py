import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (OUTPUT_DIR, CROP_SIZE, BATCH_SIZE, NUM_WORKERS, BASE_FILTERS,
                    TEACHER_EPOCHS, TEST_MODE, TEST_EPOCHS, SEED)
from dataset import get_dataloaders
from losses import CombinedSegLoss, dice_brats
from baselines.unet_baseline.model import UNetBaseline3D


def resolve_output_dir():
    if os.name == "nt" and OUTPUT_DIR.startswith("/app"):
        return os.path.abspath(os.path.join(ROOT, "..", "outputs"))
    return OUTPUT_DIR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Local BraTS root path. Overrides config DATA_ROOT.",
    )
    return parser.parse_args()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    scores = {"WT": [], "TC": [], "ET": []}
    for imgs, segs, _ in loader:
        imgs = imgs.to(device)
        segs = segs.to(device)
        out = model(imgs)
        m = dice_brats(out, segs)
        for key in scores:
            scores[key].append(m[key])
    return {k: float(np.mean(v)) for k, v in scores.items()}


def main():
    args = get_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    n_epochs = args.epochs if args.epochs is not None else (
        TEST_EPOCHS if TEST_MODE else TEACHER_EPOCHS
    )
    print(f"Training epochs: {n_epochs}")

    # Baseline-1 trains with full modalities only (missing_prob=0.0).
    tr_loader, va_loader, _ = get_dataloaders(
        data_root=args.data_root,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        crop_size=CROP_SIZE,
        missing_prob=0.0,
    )

    model = UNetBaseline3D(in_channels=4, n_classes=4, base_filters=BASE_FILTERS).to(device)
    criterion = CombinedSegLoss(alpha=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=2e-6)

    out_dir = os.path.join(resolve_output_dir(), "baselines", "unet_baseline")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, "baseline1_best.pth")
    history_path = os.path.join(out_dir, "baseline1_history.json")

    best_wt = 0.0
    history = []
    for ep in range(1, n_epochs + 1):
        model.train()
        epoch_losses = []
        for imgs, segs, _ in tqdm(tr_loader, desc=f"Baseline1 {ep}/{n_epochs}", leave=False):
            imgs = imgs.to(device)
            segs = segs.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, segs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        val = validate(model, va_loader, device)
        avg_loss = float(np.mean(epoch_losses))
        history.append({"ep": ep, "loss": avg_loss, **val})
        print(
            f"ep{ep:3d} | loss {avg_loss:.4f} | "
            f"WT {val['WT']:.4f} TC {val['TC']:.4f} ET {val['ET']:.4f}"
        )

        if val["WT"] > best_wt:
            best_wt = val["WT"]
            torch.save({"ep": ep, "best_wt": best_wt, "model_state": model.state_dict()}, best_ckpt)
            print(f"  Saved best baseline checkpoint -> {best_ckpt} (WT={best_wt:.4f})")

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best WT: {best_wt:.4f}")
    print(f"History saved -> {history_path}")


if __name__ == "__main__":
    main()
