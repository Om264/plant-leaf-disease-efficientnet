# Plant Leaf Disease Classification — EfficientNet

Fine-tunes an ImageNet-pretrained **EfficientNet** (v1 or v2, via
[`timm`](https://github.com/huggingface/pytorch-image-models)) on a custom plant-leaf image
dataset, with full training, evaluation, prediction, and Grad-CAM explainability.

This is the fourth repository in the series (after MobileNetV3, Inception v3, and ViT).

## Which EfficientNet variant is right for you?

I looked at recent (2024-2025) plant-leaf-disease papers before choosing a default. Short
answer: **`efficientnet_b3`** is the default in this repo, but which variant is actually best
for *your* project depends on your dataset size and deployment target — there is no single
universally-correct choice. Here's what the literature and the architecture itself suggest:

**The accuracy curve flattens quickly past B3.** A 2025 crop-disease paper (Scientific
Reports) benchmarking B0 through B7 found B3 gives "a significant jump over B0–B2... while B4
to B7 only improve accuracy by 2–3%, but require dramatically more computational resources"
(1.8 GFLOPs for B3 vs. up to 37 GFLOPs for B7) — and noted B4–B7 become impractical on edge
devices like a Jetson Nano or Raspberry Pi. That said, other papers (e.g. a 2020 ScienceDirect
study on PlantVillage) report B4/B5 edging out B3 by a further 1-3 points when the dataset is
large and clean — so the extra cost *can* pay off if your data supports it.

**Very high accuracy numbers (99%+) tend to come from PlantVillage-style lab datasets** —
single leaf, plain background, controlled lighting — regardless of which EfficientNet variant
is used (B0 and B4 have both individually been reported near 99% on PlantVillage-derived data
in different papers). On more realistic, field-collected images (e.g. PlantDoc), accuracy is
markedly lower across all architectures. As with the earlier repos in this series: treat a
paper's headline number as tied to *its* dataset, not as a property of the architecture alone,
and watch for the leakage risk described below.

**For edge/mobile deployment specifically**, a 2025 edge-computing comparison found MobileNetV2
edged out EfficientNetV2 on inference speed and model size on-device, while EfficientNetV2 was
stronger on multi-scale feature detection. A potato-leaf paper the same year used
**EfficientNet-Lite** (the SE-block-free, mobile/TFLite-oriented variant) specifically for
diverse/uncontrolled field conditions. If on-device deployment (not just accuracy) is a project
requirement, benchmark against the MobileNetV3 repo in this series too, or try
`tf_efficientnet_lite0`.

**EfficientNetV2** (Tan & Le, 2021) is a newer compound-scaling revision that trains faster and
is a common modern default; use `tf_efficientnetv2_s` if you have GPU headroom and want the
current standard variant rather than the original v1 line.

### Practical recommendation

| Situation | Try |
|---|---|
| Default / general-purpose (this repo's default) | `efficientnet_b3` |
| Small dataset (few hundred images/class or fewer), or CPU-bound training | `efficientnet_b0` |
| Edge/mobile deployment, real-time inference | `tf_efficientnet_lite0` (or the MobileNetV3 repo) |
| Large, clean dataset, GPU headroom, chasing max accuracy | `efficientnet_b4`, `efficientnet_b5`, or `tf_efficientnetv2_s` |
| Want the modern default over the original 2019 EfficientNet | `tf_efficientnetv2_s` |

Whichever you pick, **benchmark 2-3 variants on your own held-out test set** — the papers above
disagree with each other on B3 vs. B4 vs. B5 precisely because the right answer depends on
dataset size, image quality, and class count, all of which vary per project. This repo's
`--model` flag makes that a one-line change (see below).

## Repository layout

```
├── README.md
├── requirements.txt / requirements-dev.txt
├── train.py               # fine-tune + evaluate on the test set
├── evaluate.py              # evaluate an existing checkpoint on any labeled folder
├── predict.py                # predict on one image or a folder of images
├── gradcam.py                  # Grad-CAM heatmap for one prediction
├── plantdisease/
│   ├── data.py              # dataset discovery, stratified split, transforms
│   ├── model.py              # EfficientNet creation, checkpoint I/O
│   ├── engine.py              # train/eval loops
│   ├── metrics.py              # accuracy/precision/recall/F1/kappa/MCC/ROC-AUC + reports
│   ├── plots.py                # training curves, confusion matrices
│   └── utils.py                # seeding, device
├── scripts/
│   └── split_dataset.py      # group-aware stratified split (prevents leakage)
└── tests/
    └── test_smoke.py          # end-to-end tests: B0, EfficientNetV2-S, Grad-CAM
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## 1. Prepare your dataset

Two layouts are supported.

**A. Already split:**
```
data/
  train/  Healthy/*.jpg  Blight/*.jpg  Rust/*.jpg ...
  val/    Healthy/*.jpg  ...
  test/   Healthy/*.jpg  ...
```
(`val`/`validation`/`valid` and `test`/`testing` are all recognized.)

**B. Single folder** — `train.py` will create a stratified 70/15/15 split automatically:
```
data/
  Healthy/*.jpg
  Blight/*.jpg
  Rust/*.jpg
```

### Avoiding data leakage

If multiple images come from the same physical leaf, plant, or field plot, a naive random
split can leak near-duplicates into both train and test and inflate your reported accuracy —
a very plausible explanation for some of the 99%+ numbers in the literature. Use the
group-aware splitter instead of the automatic split whenever that's a risk:

```bash
python scripts/split_dataset.py --src raw_data --dst data/split \
    --group_regex "^(leaf\d+)_"     # example: files named leaf012_img03.jpg
```

## 2. Train

```bash
python train.py --data_dir data/split --output_dir runs/exp1 --epochs 30
```

Try a different variant with one flag, e.g.:
```bash
python train.py --data_dir data/split --output_dir runs/b0    --model efficientnet_b0
python train.py --data_dir data/split --output_dir runs/b4    --model efficientnet_b4
python train.py --data_dir data/split --output_dir runs/v2s   --model tf_efficientnetv2_s
python train.py --data_dir data/split --output_dir runs/lite0 --model tf_efficientnet_lite0
```

Key options (`python train.py --help` for the full list):

| Flag | Default | Notes |
|---|---|---|
| `--model` | `efficientnet_b3` | see the variant table above |
| `--img_size` | variant's native size | B0: 224, B3: 288, B4: 320, EfficientNetV2-S: 300, ... — picked up automatically |
| `--freeze_epochs` | `2` | epochs training only the classifier head before unfreezing |
| `--batch_size` | `32` | lower this first for B4+/V2-S/M if you hit GPU out-of-memory |
| `--class_weights` | off | use for imbalanced disease classes |
| `--patience` | `8` | early stopping on validation macro-F1 |

Outputs in `runs/exp1/`:
- `best_model.pth` — checkpoint (weights + class names + preprocessing config)
- `test_metrics.json`, `test_classification_report.txt`, `test_per_class_metrics.csv`
- `test_confusion_matrix.png` (+ normalized version), `test_predictions.csv`
- `training_curves.png`, `training_history.csv`
- `args.json` — the exact arguments used, for reproducibility

Metrics reported: accuracy, precision/recall/F1 (macro & weighted), per-class
precision/recall/F1/specificity, Cohen's kappa, Matthews correlation coefficient, ROC-AUC
(one-vs-rest macro), plus parameter count, checkpoint size, and CPU/GPU inference latency —
useful for directly comparing variants on the accuracy-vs-cost trade-off discussed above.

## 3. Evaluate on a separate labeled set

```bash
python evaluate.py --checkpoint runs/exp1/best_model.pth --test_dir data/external_test
```

## 4. Predict on new images

```bash
python predict.py --checkpoint runs/exp1/best_model.pth --input leaf.jpg
python predict.py --checkpoint runs/exp1/best_model.pth --input new_leaves/ --csv preds.csv
```

## 5. Explain a prediction (Grad-CAM)

```bash
python gradcam.py --checkpoint runs/exp1/best_model.pth --image leaf.jpg --output cam.png
```
Uses `model.conv_head` (the final 1×1 convolution before pooling) by default. Use this to
sanity-check that the model is attending to lesions rather than background, pot, or label
artifacts — a common failure mode behind inflated benchmark numbers.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```
Runs a full train → checkpoint → predict → evaluate cycle plus a Grad-CAM check on
`efficientnet_b0`, and a separate training check on `tf_efficientnetv2_s` (different internal
blocks from v1), all on tiny synthetic images with a randomly initialized model — no internet
or pretrained weights needed. Takes well under a minute on CPU.

## Reproducibility

`--seed` (default 42) seeds Python, NumPy, and PyTorch. `args.json` in every run's output
folder records the exact configuration used.

## Sources consulted

- Sangar & Rajasekar (2025), *Optimized classification of potato leaf disease using
  EfficientNet-LITE and KE-SVM*, Frontiers in Plant Science.
- *Spatial attention-guided pre-trained networks for accurate identification of crop diseases*
  (2025), Scientific Reports — B0–B7 accuracy/FLOPs comparison.
- *A fine tuned EfficientNet-B0 CNN for accurate and efficient classification of apple leaf
  diseases* (2025), Scientific Reports.
- *Plant leaf disease classification using EfficientNet deep learning model* (ScienceDirect) —
  B4/B5 vs. other EfficientNet sizes on PlantVillage.
- *Comparative and edge-hybrid modeling of EfficientNetV2 and MobileNetV2 for multi-class crop
  disease classification with statistical validation* (2025), Journal of Edge Computing.
- *Plant Leaf Disease Detection Using Deep Learning: A Multi-Dataset Approach* (2025), MDPI *J*
  — PlantDoc + web-sourced images, B0/B3 vs. ResNet50/DenseNet201 under realistic conditions.

## License

MIT — see [LICENSE](LICENSE).
