"""End-to-end smoke test on tiny synthetic data (no internet, random-init model)."""
import json

import numpy as np
from PIL import Image


def _make_dataset(root, n_classes=3, n_per_class=20, size=64):
    rng = np.random.default_rng(0)
    for c in range(n_classes):
        d = root / f"class_{c}"
        d.mkdir(parents=True)
        for i in range(n_per_class):
            arr = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
            arr[..., c] += 150  # class-dependent colour
            Image.fromarray(arr).save(d / f"img_{i}.png")


def test_train_predict_evaluate(tmp_path):
    from evaluate import main as evaluate_main
    from predict import main as predict_main
    from train import main as train_main

    data, out = tmp_path / "data", tmp_path / "out"
    _make_dataset(data)
    train_main(["--data_dir", str(data), "--output_dir", str(out),
                "--model", "efficientnet_b0", "--no_pretrained",
                "--epochs", "2", "--batch_size", "8", "--img_size", "64",
                "--workers", "0", "--freeze_epochs", "1"])

    for name in ["best_model.pth", "test_metrics.json", "test_classification_report.txt",
                 "test_confusion_matrix.png", "test_per_class_metrics.csv", "training_curves.png"]:
        assert (out / name).exists(), name
    metrics = json.loads((out / "test_metrics.json").read_text())
    assert 0.0 <= metrics["accuracy"] <= 1.0

    img = next((data / "class_0").glob("*.png"))
    predict_main(["--checkpoint", str(out / "best_model.pth"), "--input", str(img)])
    evaluate_main(["--checkpoint", str(out / "best_model.pth"), "--test_dir", str(data),
                   "--workers", "0"])
    assert (out / "eval_metrics.json").exists()


def test_gradcam(tmp_path):
    from gradcam import main as gradcam_main
    from train import main as train_main

    data, out = tmp_path / "data", tmp_path / "out"
    _make_dataset(data, n_per_class=12)
    train_main(["--data_dir", str(data), "--output_dir", str(out),
                "--model", "efficientnet_b0", "--no_pretrained",
                "--epochs", "1", "--batch_size", "8", "--img_size", "64", "--workers", "0"])

    img = next((data / "class_0").glob("*.png"))
    cam_out = out / "cam.png"
    gradcam_main(["--checkpoint", str(out / "best_model.pth"), "--image", str(img),
                  "--output", str(cam_out)])
    assert cam_out.exists()


def test_efficientnetv2_variant(tmp_path):
    """Also verify an EfficientNetV2 variant trains (different block/scaling internals)."""
    from train import main as train_main

    data, out = tmp_path / "data", tmp_path / "out"
    _make_dataset(data, n_classes=2, n_per_class=10)
    train_main(["--data_dir", str(data), "--output_dir", str(out),
                "--model", "tf_efficientnetv2_s", "--no_pretrained",
                "--epochs", "1", "--batch_size", "4", "--img_size", "96", "--workers", "0"])
    assert (out / "best_model.pth").exists()
