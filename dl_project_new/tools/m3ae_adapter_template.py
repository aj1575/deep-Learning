"""
Template adapter for evaluating an external M3AE model with evaluate_all_models.py.

Implement these two functions:
  - load_model(ckpt_path, device, **kwargs) -> (model, state_dict_or_meta)
  - predict(model, imgs_masked, modality_mask) -> logits [B, C, H, W, D]
"""

import torch


def load_model(ckpt_path, device, **kwargs):
    """
    Load your M3AE model and checkpoint.
    Return (model, state), where state can be checkpoint metadata dict.
    """
    raise NotImplementedError("Implement model construction and checkpoint loading for your M3AE codebase.")


def predict(model, imgs_masked, modality_mask):
    """
    Return segmentation logits with shape [B, num_classes, H, W, D].
    modality_mask is a float tensor [B,4] containing current modality availability.
    """
    with torch.no_grad():
        out = model(imgs_masked, modality_mask)
        if isinstance(out, tuple):
            out = out[0]
        return out
