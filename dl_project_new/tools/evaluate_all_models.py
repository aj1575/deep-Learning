import os
import sys
import json
import argparse
import importlib.util
import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import OUTPUT_DIR, CROP_SIZE, BASE_FILTERS, LATENT_DIM
from dataset import get_dataloaders
from losses import dice_brats
from models.brainsegnet import BrainSegNet
from baselines.unet_baseline.model import UNetBaseline3D


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


def resolve_output_dir():
    if os.name == "nt" and OUTPUT_DIR.startswith("/app"):
        return os.path.abspath(os.path.join(ROOT, "..", "outputs"))
    return OUTPUT_DIR


def run_eval(model, loader, device, predict_fn, is_brainsegnet=False):
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
                if is_brainsegnet:
                    mk = torch.tensor([mask_vals], dtype=torch.float32, device=device).expand(imgs.shape[0], -1)
                    out = predict_fn(model, imgs_masked, mk)
                else:
                    out = predict_fn(model, imgs_masked, None)
            m = dice_brats(out, segs)
            for key in d:
                d[key].append(m[key])
        avg = {k: float(np.mean(d[k])) for k in d}
        results[name] = avg
        print(f"    WT {avg['WT']:.4f} TC {avg['TC']:.4f} ET {avg['ET']:.4f}")

    results["MEAN"] = {
        k: float(np.mean([results[name][k] for name, _ in ALL_COMBOS]))
        for k in ["WT", "TC", "ET"]
    }
    return results


def load_baseline_ckpt(path, device):
    model = UNetBaseline3D(in_channels=4, n_classes=4, base_filters=BASE_FILTERS).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    return model, state


def predict_unet(model, imgs_masked, _mask):
    return model(imgs_masked)


def load_brainsegnet_ckpt(path, device):
    state = torch.load(path, map_location=device)
    state_dict = state["model_state"] if isinstance(state, dict) and "model_state" in state else state

    has_gan_weights = any(k.startswith("generator.") or k.startswith("discriminator.") for k in state_dict.keys())
    model = BrainSegNet(
        base_filters=BASE_FILTERS,
        crop_size=CROP_SIZE,
        latent_dim=LATENT_DIM,
        use_gan=has_gan_weights,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)

    wrapped_state = state if isinstance(state, dict) else {"model_state": state_dict}
    wrapped_state["detected_use_gan"] = has_gan_weights
    return model, wrapped_state


def predict_brainsegnet(model, imgs_masked, mask_tensor):
    out = model(imgs_masked, mask_tensor, training=False)
    if isinstance(out, tuple):
        out = out[0]
    return out


def load_protocol(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_value(arg_value, protocol, key, default=None):
    if arg_value is not None:
        return arg_value
    return protocol.get(key, default)


def load_external_adapter(adapter_path):
    spec = importlib.util.spec_from_file_location("external_model_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load adapter module: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for fn_name in ("load_model", "predict"):
        if not hasattr(module, fn_name):
            raise AttributeError(f"Adapter must implement `{fn_name}` in {adapter_path}")
    return module


def get_args():
    p = argparse.ArgumentParser(description="Evaluate Baseline1, Baseline2, BrainSegNet on one dataset (no training).")
    p.add_argument("--data-root", required=True, help="Dataset root containing case folders.")
    p.add_argument("--protocol-config", default=None, help="Optional JSON file with locked comparison settings.")
    p.add_argument("--split-file", default=None, help="Optional JSON split file with train/val/test patient IDs.")
    p.add_argument(
        "--baseline1-ckpt",
        default=None,
        help="Path to baseline1 checkpoint. Default: outputs/baselines/unet_baseline/checkpoints/baseline1_best.pth",
    )
    p.add_argument(
        "--baseline2-ckpt",
        default=None,
        help="Path to baseline2 checkpoint. Default: outputs/baselines/unet_baseline/checkpoints/baseline2_best.pth",
    )
    p.add_argument(
        "--brainsegnet-ckpt",
        default=None,
        help="Path to BrainSegNet student checkpoint (.pth). Required if --skip-brainsegnet is not used.",
    )
    p.add_argument("--skip-brainsegnet", action="store_true", help="Skip BrainSegNet if checkpoint is unavailable.")
    p.add_argument("--m3ae-ckpt", default=None, help="Path to M3AE checkpoint (requires --m3ae-adapter).")
    p.add_argument(
        "--m3ae-adapter",
        default=None,
        help="Python file that implements load_model(ckpt, device, **kwargs) and predict(model, imgs, mask).",
    )
    return p.parse_args()


def main():
    args = get_args()
    protocol = load_protocol(args.protocol_config)
    out_root = resolve_output_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    split_file = resolve_value(args.split_file, protocol, "split_file", None)

    b1_ckpt = resolve_value(args.baseline1_ckpt, protocol, "baseline1_ckpt", None) or os.path.join(
        out_root, "baselines", "unet_baseline", "checkpoints", "baseline1_best.pth"
    )
    b2_ckpt = resolve_value(args.baseline2_ckpt, protocol, "baseline2_ckpt", None) or os.path.join(
        out_root, "baselines", "unet_baseline", "checkpoints", "baseline2_best.pth"
    )
    bsn_ckpt = resolve_value(args.brainsegnet_ckpt, protocol, "brainsegnet_ckpt", None)
    m3ae_ckpt = resolve_value(args.m3ae_ckpt, protocol, "m3ae_ckpt", None)
    m3ae_adapter = resolve_value(args.m3ae_adapter, protocol, "m3ae_adapter", None)

    if not os.path.exists(b1_ckpt):
        raise FileNotFoundError(f"Baseline1 checkpoint not found: {b1_ckpt}")
    if not os.path.exists(b2_ckpt):
        raise FileNotFoundError(f"Baseline2 checkpoint not found: {b2_ckpt}")
    if not args.skip_brainsegnet:
        if not bsn_ckpt:
            raise ValueError("Provide --brainsegnet-ckpt or use --skip-brainsegnet.")
        if not os.path.exists(bsn_ckpt):
            raise FileNotFoundError(f"BrainSegNet checkpoint not found: {bsn_ckpt}")
    if m3ae_ckpt is not None:
        if not m3ae_adapter:
            raise ValueError("Provide --m3ae-adapter when using --m3ae-ckpt.")
        if not os.path.exists(m3ae_ckpt):
            raise FileNotFoundError(f"M3AE checkpoint not found: {m3ae_ckpt}")
        if not os.path.exists(m3ae_adapter):
            raise FileNotFoundError(f"M3AE adapter not found: {m3ae_adapter}")

    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        crop_size=CROP_SIZE,
        missing_prob=0.0,
        split_file=split_file,
    )

    all_results = {}

    print("\n=== Evaluating Baseline 1 ===")
    b1_model, b1_state = load_baseline_ckpt(b1_ckpt, device)
    print(f"Loaded: {b1_ckpt} | best_wt={b1_state.get('best_wt', -1):.4f}")
    all_results["Baseline1"] = run_eval(
        b1_model, test_loader, device, predict_fn=predict_unet, is_brainsegnet=False
    )

    print("\n=== Evaluating Baseline 2 ===")
    b2_model, b2_state = load_baseline_ckpt(b2_ckpt, device)
    print(f"Loaded: {b2_ckpt} | best_wt={b2_state.get('best_wt', -1):.4f}")
    all_results["Baseline2"] = run_eval(
        b2_model, test_loader, device, predict_fn=predict_unet, is_brainsegnet=False
    )

    if not args.skip_brainsegnet:
        print("\n=== Evaluating BrainSegNet ===")
        bsn_model, bsn_state = load_brainsegnet_ckpt(bsn_ckpt, device)
        print(
            f"Loaded: {bsn_ckpt} | best_wt={bsn_state.get('best_wt', -1):.4f} "
            f"| use_gan={bsn_state.get('detected_use_gan')}"
        )
        all_results["BrainSegNet"] = run_eval(
            bsn_model, test_loader, device, predict_fn=predict_brainsegnet, is_brainsegnet=True
        )

    if m3ae_ckpt is not None:
        print("\n=== Evaluating M3AE ===")
        adapter = load_external_adapter(m3ae_adapter)
        m3ae_model, m3ae_state = adapter.load_model(
            m3ae_ckpt,
            device,
            crop_size=CROP_SIZE,
            base_filters=BASE_FILTERS,
            latent_dim=LATENT_DIM,
        )
        print(f"Loaded: {m3ae_ckpt}")
        if isinstance(m3ae_state, dict) and "best_wt" in m3ae_state:
            print(f"M3AE best_wt={m3ae_state['best_wt']:.4f}")
        all_results["M3AE"] = run_eval(
            m3ae_model, test_loader, device, predict_fn=adapter.predict, is_brainsegnet=True
        )

    print("\n=== Mean Metrics Summary ===")
    for model_name, r in all_results.items():
        m = r["MEAN"]
        print(f"{model_name:12s} -> WT {m['WT']:.4f} | TC {m['TC']:.4f} | ET {m['ET']:.4f}")

    out_path = os.path.join(out_root, "all_models_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined results -> {out_path}")


if __name__ == "__main__":
    main()
