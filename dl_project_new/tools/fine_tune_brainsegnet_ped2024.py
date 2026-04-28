import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import BASE_FILTERS, CROP_SIZE, LATENT_DIM, OUTPUT_DIR
from dataset import get_dataloaders
from losses import DeepSupervisionLoss, dice_brats
from models.brainsegnet import BrainSegNet


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune BrainSegNet on PED2024 in backup workspace.")
    p.add_argument("--data-root", required=True, help="PED2024 converted BraTS-style root.")
    p.add_argument("--split-file", required=True, help="Split JSON containing train/val/test lists.")
    p.add_argument("--init-ckpt", required=True, help="Initial checkpoint (.pth) to fine-tune from.")
    p.add_argument("--epochs", type=int, default=20, help="Fine-tuning epochs.")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--missing-prob", type=float, default=0.5, help="Modality dropout probability during train.")
    p.add_argument("--out-dir", default=None, help="Output folder for backup run artifacts.")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    scores = {"WT": [], "TC": [], "ET": []}
    for imgs, segs, masks in loader:
        imgs, segs, masks = imgs.to(device), segs.to(device), masks.to(device)
        out = model(imgs, masks, training=False)
        m = dice_brats(out, segs)
        for k in scores:
            scores[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in scores.items()}


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(OUTPUT_DIR, "ped2024_finetune_from_student")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    state = torch.load(args.init_ckpt, map_location=device)
    state_dict = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    has_gan = any(k.startswith("generator.") or k.startswith("discriminator.") for k in state_dict.keys())

    model = BrainSegNet(
        base_filters=BASE_FILTERS,
        crop_size=CROP_SIZE,
        latent_dim=LATENT_DIM,
        use_gan=has_gan,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded init checkpoint: {args.init_ckpt}")
    print(f"Detected use_gan: {has_gan}")

    tr_loader, va_loader, _ = get_dataloaders(
        data_root=args.data_root,
        crop_size=CROP_SIZE,
        missing_prob=args.missing_prob,
        split_file=args.split_file,
    )
    print(f"Using split file: {args.split_file}")

    criterion = DeepSupervisionLoss(0.7)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 1e-6)

    best_wt = 0.0
    history = []
    best_path = os.path.join(out_dir, "student_ped2024_finetuned_best.pth")

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for imgs, segs, masks in tqdm(tr_loader, desc=f"FineTune {ep}/{args.epochs}", leave=False):
            imgs, segs, masks = imgs.to(device), segs.to(device), masks.to(device)
            optimizer.zero_grad()

            main_out, aux3, aux2, kl, _, _ = model(imgs, masks, training=True)
            loss = criterion(main_out, aux3, aux2, segs) + 0.05 * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        val = validate(model, va_loader, device)
        avg_loss = float(np.mean(losses)) if losses else 0.0
        row = {"ep": ep, "loss": avg_loss, **val}
        history.append(row)
        print(
            f"ep{ep:03d} | loss {avg_loss:.4f} | "
            f"WT {val['WT']:.4f} TC {val['TC']:.4f} ET {val['ET']:.4f}"
        )

        if val["WT"] > best_wt:
            best_wt = val["WT"]
            torch.save({"ep": ep, "best_wt": best_wt, "model_state": model.state_dict()}, best_path)
            print(f"  Saved best -> {best_path} (WT={best_wt:.4f})")

    hist_path = os.path.join(out_dir, "student_ped2024_finetune_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history -> {hist_path}")
    print(f"Best WT -> {best_wt:.4f}")


if __name__ == "__main__":
    main()
