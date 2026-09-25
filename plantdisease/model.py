"""EfficientNet (v1 and v2) creation, checkpoint I/O.

EfficientNet is a standard convolutional network - unlike Inception v3 it has
no auxiliary head, and unlike a ViT it has ordinary spatial conv feature maps,
so it slots into the same training loop as MobileNetV3 and supports Grad-CAM
directly (via model.conv_head, the final 1x1 conv before global pooling).

What makes EfficientNet specifically "efficient" is compound scaling: depth,
width, and input resolution are all scaled together (B0 -> B7) instead of
independently, which is why - unlike MobileNetV3 or a ViT - each size variant
also has its own native input resolution (B0: 224, B3: 288, B4: 320, ...).
resolve_data_config() picks the right one up automatically per variant.
"""
from typing import Optional

import timm
import torch
from timm.data import resolve_data_config


def create_model(name: str = "efficientnet_b3", num_classes: int = 1000,
                 pretrained: bool = True, dropout: float = 0.2):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)


def data_config(model, img_size: Optional[int] = None) -> dict:
    """Input size / mean / std the pretrained weights expect (variant-specific for EfficientNet)."""
    cfg = resolve_data_config({}, model=model)
    return {
        "img_size": int(img_size or cfg["input_size"][1]),
        "mean": [float(x) for x in cfg["mean"]],
        "std": [float(x) for x in cfg["std"]],
    }


def set_backbone_frozen(model, frozen: bool) -> None:
    """Freeze everything except the classification head (and conv_head if present)."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    if hasattr(model, "conv_head"):
        head_ids |= {id(p) for p in model.conv_head.parameters()}
    for p in model.parameters():
        p.requires_grad = (not frozen) or (id(p) in head_ids)


def save_checkpoint(path, model, model_name, class_names, cfg, extra=None):
    torch.save({
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "data_cfg": cfg,
        "extra": extra or {},
    }, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = create_model(ckpt["model_name"], len(ckpt["class_names"]), pretrained=False, dropout=0.0)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
