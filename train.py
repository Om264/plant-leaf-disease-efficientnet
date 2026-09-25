"""Fine-tune a timm EfficientNet on a plant-leaf disease dataset, then evaluate on the test set.

Example:
    python train.py --data_dir data/my_leaves --epochs 30 --output_dir runs/exp1

Choosing a variant (see README.md "Which EfficientNet?" for the full reasoning):
    - efficientnet_b0 / tf_efficientnet_lite0  : smallest, fastest, best for edge/mobile
                                                  deployment or very small datasets
    - efficientnet_b3   (this script's default) : accuracy/compute sweet spot reported by
                                                  several recent plant-disease papers - most
                                                  of the gain over B0 with far less cost than
                                                  B4-B7
    - efficientnet_b4 / b5                       : a further 1-3% accuracy in some papers, at
                                                  roughly 2x the compute of B3 - only worth it
                                                  with a large, clean dataset and GPU headroom
    - tf_efficientnetv2_s                        : EfficientNetV2 (Tan & Le 2021); trains
                                                  faster than v1 and often matches or beats
                                                  B4/B5 accuracy at similar cost - a strong
                                                  alternative default if you have a decent GPU
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from plantdisease.data import build_datasets, build_transforms, make_loaders
from plantdisease.engine import measure_latency, predict_loader, train_one_epoch
from plantdisease.metrics import compute_metrics, save_evaluation
from plantdisease.model import (create_model, data_config, save_checkpoint,
                                set_backbone_frozen)
from plantdisease.plots import plot_history
from plantdisease.utils import get_device, set_seed


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="dataset root (ImageFolder layout)")
    ap.add_argument("--output_dir", default="runs/exp1")
    ap.add_argument("--model", default="efficientnet_b3",
                    help="timm EfficientNet name: efficientnet_b0..b7, tf_efficientnet_lite0..4, "
                         "tf_efficientnetv2_s/m/l, tf_efficientnetv2_b0..b3")
    ap.add_argument("--no_pretrained", action="store_true", help="random init (offline / tests)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32,
                    help="larger variants (B4+) or EfficientNetV2-S/M need a smaller batch at "
                         "their native resolution if you hit GPU out-of-memory")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--img_size", type=int, default=None,
                    help="default: the model variant's own native size (224 for B0, 288 for "
                         "B3, 320 for B4, 300 for EfficientNetV2-S, ...)")
    ap.add_argument("--freeze_epochs", type=int, default=2,
                    help="epochs to train only the classifier head before unfreezing")
    ap.add_argument("--patience", type=int, default=8, help="early-stopping patience (val macro-F1)")
    ap.add_argument("--class_weights", action="store_true", help="weight loss for imbalanced data")
    ap.add_argument("--val_size", type=float, default=0.15, help="used if no val folder exists")
    ap.add_argument("--test_size", type=float, default=0.15, help="used for single-folder data")
    ap.add_argument("--workers", type=int, default=4, help="use 0 on Windows if you get errors")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Device: {device}")

    # ---- data ----------------------------------------------------------------------
    bundle = build_datasets(args.data_dir, args.val_size, args.test_size, args.seed)
    class_names = bundle.class_names
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train/Val/Test: {len(bundle.train)}/{len(bundle.val)}/{len(bundle.test)}")
    (out / "class_names.json").write_text(json.dumps(class_names, indent=2))

    # ---- model ---------------------------------------------------------------------
    model = create_model(args.model, num_classes, pretrained=not args.no_pretrained,
                         dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | parameters: {n_params/1e6:.2f} M")

    cfg = data_config(model, args.img_size)
    print(f"Input resolution: {cfg['img_size']}px")
    train_tf, eval_tf = build_transforms(cfg)
    bundle.set_transforms(train_tf, eval_tf)
    train_loader, val_loader, test_loader = make_loaders(
        bundle, args.batch_size, args.workers, device.type == "cuda")

    # ---- optimisation --------------------------------------------------------------
    weight = None
    if args.class_weights:
        counts = np.bincount(bundle.train_labels, minlength=num_classes)
        w = counts.sum() / (num_classes * np.maximum(counts, 1))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ---- training loop -------------------------------------------------------------
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]}
    best_f1, bad_epochs = -1.0, 0
    best_path = out / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        set_backbone_frozen(model, frozen)
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                          scaler, device, use_amp)
        scheduler.step()

        probs, labels, val_loss = predict_loader(model, val_loader, device, criterion)
        vm = compute_metrics(labels, probs, num_classes)
        for k, v in zip(hist, [tr_loss, tr_acc, val_loss, vm["accuracy"], vm["f1_macro"]]):
            hist[k].append(v)
        print(f"Epoch {epoch:03d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {vm['accuracy']:.4f} F1 {vm['f1_macro']:.4f}"
              f"{'  [head only]' if frozen else ''}")

        if vm["f1_macro"] > best_f1:
            best_f1, bad_epochs = vm["f1_macro"], 0
            save_checkpoint(best_path, model, args.model, class_names, cfg,
                            extra={"epoch": epoch, "val_f1_macro": best_f1})
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no val macro-F1 improvement for {args.patience} epochs.")
                break

    pd.DataFrame(hist).to_csv(out / "training_history.csv", index=False)
    plot_history(hist, out / "training_curves.png")

    # ---- final test evaluation with the best checkpoint ----------------------------
    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    probs, labels, _ = predict_loader(model, test_loader, device)
    extra = {
        "best_epoch": int(torch.load(best_path, map_location="cpu")["extra"]["epoch"]),
        "params_millions": n_params / 1e6,
        "model_size_mb": best_path.stat().st_size / 1e6,
        "latency_ms_per_image_cpu": measure_latency(model, torch.device("cpu"), cfg["img_size"], runs=30),
    }
    if device.type == "cuda":
        extra["latency_ms_per_image_gpu"] = measure_latency(model, device, cfg["img_size"])
    save_evaluation(out, labels, probs, class_names, tag="test",
                    paths=[p for p, _ in bundle.test.samples], extra=extra)
    print(f"All outputs saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
