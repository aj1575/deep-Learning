"""
Adapter for official M3AE repo (https://github.com/ccarliu/m3ae).

This adapter bridges official M3AE outputs (ET/TC/WT heads) into this project's
4-class logit convention used by dice_brats():
  class 0: background
  class 1: tumor core non-enhancing (TC minus ET)
  class 2: edema-like (WT minus TC)
  class 3: enhancing tumor (ET)
"""

import os
import sys
from pathlib import Path

import torch


def _resolve_m3ae_root(explicit_root=None):
    if explicit_root:
        root = Path(explicit_root).resolve()
        if (root / "model" / "Unet.py").exists():
            return root
        raise FileNotFoundError(f"Invalid m3ae_root: {root}")

    # Default: sibling folder at repository root.
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # BrainSegNet/
    default = repo_root / "m3ae_official"
    if (default / "model" / "Unet.py").exists():
        return default
    raise FileNotFoundError(
        "Could not locate official M3AE repo. Expected at: "
        f"{default}. Pass m3ae_root in protocol config if different."
    )


def _canonical_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def load_model(ckpt_path, device, **kwargs):
    m3ae_root = _resolve_m3ae_root(kwargs.get("m3ae_root"))
    if str(m3ae_root) not in sys.path:
        sys.path.insert(0, str(m3ae_root))

    # Import from official repo after path injection.
    from model.Unet import Unet_missing, proj

    patch_shape = int(kwargs.get("crop_size", 128))
    mdp = int(kwargs.get("mdp", 3))
    init_channels = int(kwargs.get("init_channels", 16))

    model = Unet_missing(
        input_shape=[patch_shape, patch_shape, patch_shape],
        out_channels=3,  # ET/TC/WT heads in official code
        init_channels=init_channels,
        pre_train=False,
        mdp=mdp,
        patch_shape=patch_shape,
    )
    # Keep on CPU inside module per official code behavior.
    model.raw_input = proj(torch.ones((1, 4, patch_shape, patch_shape, patch_shape))).cpu()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _canonical_state_dict(ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, ckpt


def _ensure_patch_shape(model, patch_shape):
    # Official model stores a cached token tensor built for patch_shape.
    if int(getattr(model, "patch_shape", -1)) == int(patch_shape):
        return
    from model.Unet import proj

    model.patch_shape = int(patch_shape)
    model.raw_input = proj(torch.ones((1, 4, patch_shape, patch_shape, patch_shape))).cpu()


def _m3ae_to_4class_logits(m3ae_logits):
    """
    Convert official 3-head output [ET, TC, WT] to 4-class scores.
    This preserves hierarchical constraints approximately:
      ET ⊆ TC ⊆ WT.
    """
    probs = torch.sigmoid(m3ae_logits)
    et = probs[:, 0:1]
    tc = probs[:, 1:2]
    wt = probs[:, 2:3]

    c3 = et
    c1 = torch.clamp(tc - et, min=0.0, max=1.0)
    c2 = torch.clamp(wt - tc, min=0.0, max=1.0)
    c0 = torch.clamp(1.0 - wt, min=0.0, max=1.0)
    return torch.cat([c0, c1, c2, c3], dim=1)


def predict(model, imgs_masked, modality_mask):
    """
    Args:
      imgs_masked: [B,4,H,W,D]
      modality_mask: [B,4] float mask, 1=present, 0=missing
    Returns:
      logits-like tensor [B,4,H,W,D] compatible with dice_brats() argmax logic.
    """
    d, h, w = imgs_masked.shape[-3:]
    if d != h or h != w:
        raise ValueError(f"M3AE adapter expects cubic patches. Got shape {imgs_masked.shape[-3:]}")

    _ensure_patch_shape(model, d)

    mask_row = modality_mask[0].detach().cpu().tolist() if modality_mask is not None else [1, 1, 1, 1]
    missing_modalities = [i for i, present in enumerate(mask_row) if present < 0.5]
    model.mask_modal = missing_modalities

    location = ((0, d), (0, h), (0, w))
    with torch.no_grad():
        out = model(imgs_masked, location=location)
        if isinstance(out, tuple):
            out = out[0]
        return _m3ae_to_4class_logits(out)
