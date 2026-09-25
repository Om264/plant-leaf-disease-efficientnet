"""Grad-CAM heat-map: which part of the leaf drove the prediction.

Example:
    python gradcam.py --checkpoint runs/exp1/best_model.pth --image leaf.jpg --output cam.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from plantdisease.data import build_eval_transform
from plantdisease.model import load_checkpoint
from plantdisease.utils import get_device


class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.acts, self.grads = model, None, None
        target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
            idx = int(logits.argmax(1)) if class_idx is None else class_idx
            logits[0, idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), idx, probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/exp1/best_model.pth")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default="gradcam.png")
    ap.add_argument("--class_idx", type=int, default=None, help="default: predicted class")
    ap.add_argument("--target_layer", default=None,
                    help="module name, default: model.conv_head (final 1x1 conv before pooling)")
    args = ap.parse_args(argv)

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    class_names, cfg = ckpt["class_names"], ckpt["data_cfg"]

    layer = model.get_submodule(args.target_layer) if args.target_layer else model.conv_head
    cam_fn = GradCAM(model, layer)

    img = Image.open(args.image).convert("RGB")
    x = build_eval_transform(cfg)(img).unsqueeze(0).to(device)
    cam, idx, probs = cam_fn(x, args.class_idx)

    size = cfg["img_size"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(img.resize((size, size)))
    ax[0].set_title("Input")
    ax[1].imshow(img.resize((size, size)))
    ax[1].imshow(cam, cmap="jet", alpha=0.45)
    ax[1].set_title(f"{class_names[idx]}\n({probs[idx]:.1%})")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved {args.output} | predicted: {class_names[idx]} ({probs[idx]:.4f})")


if __name__ == "__main__":
    main()
