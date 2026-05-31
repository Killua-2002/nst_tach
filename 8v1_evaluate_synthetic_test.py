"""
8v1_evaluate_synthetic_test.py
Evaluate best NST segmentation model on synthetic labelled test data (dataset/test).

Mục đích:
- Bỏ qua ảnh real overlap_raw nếu domain chưa khớp.
- Dùng split test có ground-truth từ pipeline 3v/4v/5v để đo accuracy thật.
- Load best model đã train, predict A/B/C, tính Dice/IoU/Pixel Accuracy.
- Xuất CSV/JSON và ảnh visualize so sánh GT vs Pred.

Chạy trong project root/Colab:
    python -u 8v1_evaluate_synthetic_test.py \
      --dataset-dir /content/dataset \
      --results-dir /content/results \
      --split test \
      --sample-ratio 1.0 \
      --visualize 30

Output:
    results/synthetic_test_eval/metrics.json
    results/synthetic_test_eval/per_image_metrics.csv
    results/synthetic_test_eval/visualizations/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Giữ terminal sạch hơn
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE = 256
THRESHOLDS = np.array([0.50, 0.50, 0.40], dtype=np.float32)  # A, B, C
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG")


def find_images(folder: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMG_EXTS:
        files.extend(folder.glob(ext))
    return sorted(set(files))


def split_dirs(dataset_dir: Path, split: str) -> Tuple[Path, Path, Path, Path]:
    base = dataset_dir / split
    return base / "images", base / "masks_A", base / "masks_B", base / "masks_C"


def list_samples(dataset_dir: Path, split: str) -> List[Tuple[Path, Path, Path, Path]]:
    image_dir, mask_a_dir, mask_b_dir, mask_c_dir = split_dirs(dataset_dir, split)
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")

    samples: List[Tuple[Path, Path, Path, Path]] = []
    for img_path in find_images(image_dir):
        ma = mask_a_dir / img_path.name
        mb = mask_b_dir / img_path.name
        mc = mask_c_dir / img_path.name
        if ma.exists() and mb.exists() and mc.exists():
            samples.append((img_path, ma, mb, mc))
        else:
            print(f"[WARN] Missing mask for {img_path.name}, skipped.")

    if not samples:
        raise FileNotFoundError(f"No valid labelled samples found in {dataset_dir}/{split}")
    return samples


def choose_samples(samples: Sequence[Tuple[Path, Path, Path, Path]], sample_ratio: float, max_samples: int, seed: int):
    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)

    if sample_ratio <= 0:
        raise ValueError("--sample-ratio must be > 0")
    n = len(samples) if sample_ratio >= 1.0 else max(1, int(round(len(samples) * sample_ratio)))
    if max_samples > 0:
        n = min(n, max_samples)
    return samples[:n]


def load_gray_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    # Dataset sau 4v thường đã là 256, nhưng resize lại để chắc shape đúng với model.
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[..., None]


def load_mask(path: Path) -> np.ndarray:
    m = Image.open(path).convert("L")
    if m.size != (IMG_SIZE, IMG_SIZE):
        m = m.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = (np.asarray(m, dtype=np.uint8) > 127).astype(np.float32)
    return arr[..., None]


def load_batch(batch_samples: Sequence[Tuple[Path, Path, Path, Path]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    xs, ys, names = [], [], []
    for img_path, ma, mb, mc in batch_samples:
        x = load_gray_image(img_path)
        y = np.concatenate([load_mask(ma), load_mask(mb), load_mask(mc)], axis=-1)
        xs.append(x)
        ys.append(y)
        names.append(img_path.name)
    return np.stack(xs, axis=0).astype(np.float32), np.stack(ys, axis=0).astype(np.float32), names


def model_candidates(results_dir: Path) -> List[Path]:
    return [
        results_dir / "best_for_apply_inference.keras",
        results_dir / "best_for_apply.keras",
        results_dir / "best_teacher.keras",
        results_dir / "best_student.keras",
        results_dir / "final_teacher_inference.keras",
        results_dir / "final_teacher.keras",
        results_dir / "final_student_inference.keras",
        results_dir / "final_student.keras",
    ]


def resolve_model_path(results_dir: Path, model_path: str) -> Path:
    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        return p
    for p in model_candidates(results_dir):
        if p.exists():
            return p
    raise FileNotFoundError(
        "No model found. Expected one of: " + ", ".join(str(p) for p in model_candidates(results_dir))
    )


def load_model_for_inference(path: Path) -> keras.Model:
    print(f"[MODEL] Loading: {path} ({path.stat().st_size/1024/1024:.1f} MB)")
    try:
        return keras.models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        return keras.models.load_model(path, compile=False)


def binarize_pred(prob: np.ndarray) -> np.ndarray:
    return (prob[..., :3] > THRESHOLDS.reshape(1, 1, 1, 3)).astype(np.uint8)


def metrics_one(y_true: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    # y_true/y_pred_bin shape: [H,W,3] hoặc [1,H,W,3]
    if y_true.ndim == 3:
        y_true = y_true[None, ...]
    if y_pred_bin.ndim == 3:
        y_pred_bin = y_pred_bin[None, ...]

    yt = y_true.astype(bool)
    yp = y_pred_bin.astype(bool)
    out: Dict[str, float] = {}
    dice_vals, iou_vals = [], []
    names = ["A", "B", "C"]

    for i, name in enumerate(names):
        t = yt[..., i]
        p = yp[..., i]
        inter = np.logical_and(t, p).sum(axis=(1, 2)).astype(np.float64)
        denom = t.sum(axis=(1, 2)) + p.sum(axis=(1, 2))
        union = np.logical_or(t, p).sum(axis=(1, 2)).astype(np.float64)

        dice = np.where(denom == 0, 1.0, (2.0 * inter) / np.maximum(denom, 1))
        iou = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
        out[f"dice_{name}"] = float(np.mean(dice))
        out[f"iou_{name}"] = float(np.mean(iou))
        dice_vals.append(out[f"dice_{name}"])
        iou_vals.append(out[f"iou_{name}"])

    out["mean_dice_ABC"] = float(np.mean(dice_vals))
    out["mean_iou_ABC"] = float(np.mean(iou_vals))
    out["pixel_acc_ABC"] = float(np.mean(yt == yp))
    out["accuracy_percent_main_diceABC"] = out["mean_dice_ABC"] * 100.0
    out["pixel_accuracy_percent"] = out["pixel_acc_ABC"] * 100.0
    return out


def weighted_average(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (float, int))]
    out = {k: float(np.mean([float(r[k]) for r in rows])) for k in keys}
    out["samples"] = float(len(rows))
    return out


def make_overlay(gray: np.ndarray, mask_bin: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    # gray [H,W] 0..1, mask_bin [H,W,3] bool/int
    base = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    colors = np.zeros_like(base)
    a = mask_bin[..., 0].astype(bool)
    b = mask_bin[..., 1].astype(bool)
    c = mask_bin[..., 2].astype(bool)

    # A đỏ, B xanh lá, C vàng
    colors[a] = np.array([1.0, 0.0, 0.0])
    colors[b] = np.array([0.0, 1.0, 0.0])
    colors[c] = np.array([1.0, 1.0, 0.0])

    any_mask = np.any(mask_bin.astype(bool), axis=-1)
    out = base.copy()
    out[any_mask] = (1 - alpha) * base[any_mask] + alpha * colors[any_mask]
    return np.clip(out, 0, 1)


def save_visualization(out_path: Path, name: str, x: np.ndarray, y_true: np.ndarray, prob: np.ndarray, row_metrics: Dict[str, float]):
    pred_bin = binarize_pred(prob[None, ...])[0]
    true_bin = (y_true > 0.5).astype(np.uint8)
    gray = x[..., 0]

    gt_overlay = make_overlay(gray, true_bin)
    pred_overlay = make_overlay(gray, pred_bin)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(
        f"{name} | DiceABC={row_metrics['mean_dice_ABC']:.4f} "
        f"A={row_metrics['dice_A']:.3f} B={row_metrics['dice_B']:.3f} C={row_metrics['dice_C']:.3f}",
        fontsize=12,
    )
    panels = [
        (gray, "Input", "gray"),
        (gt_overlay, "GT overlay (đáp án)", None),
        (pred_overlay, "Pred overlay", None),
        (np.abs(true_bin.astype(float) - pred_bin.astype(float)).max(axis=-1), "Error map", "magma"),
        (true_bin[..., 0], "GT A", "gray"),
        (pred_bin[..., 0], "Pred A", "gray"),
        (true_bin[..., 1], "GT B", "gray"),
        (pred_bin[..., 1], "Pred B", "gray"),
    ]
    for ax, (img, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if cmap else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Thêm C ở một file phụ? Để giữ layout 2x4 vừa gọn, lưu thêm C vào góc title/error đã có.
    # Tạo inset nhỏ bằng text để biết C metric.
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_c_visualization(out_path: Path, name: str, x: np.ndarray, y_true: np.ndarray, prob: np.ndarray, row_metrics: Dict[str, float]):
    pred_bin = binarize_pred(prob[None, ...])[0]
    true_bin = (y_true > 0.5).astype(np.uint8)
    gray = x[..., 0]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    fig.suptitle(f"{name} | Overlap C Dice={row_metrics['dice_C']:.4f}", fontsize=11)
    panels = [
        (gray, "Input", "gray"),
        (true_bin[..., 2], "GT C / overlap", "gray"),
        (pred_bin[..., 2], "Pred C / overlap", "gray"),
        (np.abs(true_bin[..., 2].astype(float) - pred_bin[..., 2].astype(float)), "C error", "magma"),
    ]
    for ax, (img, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_csv(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model-path", type=str, default="", help="Optional explicit .keras file. Default auto-picks best_for_apply/best_teacher.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="1.0 = all samples in split. 0.1 = 10% of split.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = no cap.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", type=int, default=30, help="Number of visual comparison images to save.")
    parser.add_argument("--out-dir", type=str, default="", help="Default: results/synthetic_test_eval")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "synthetic_test_eval"
    viz_dir = out_dir / "visualizations"
    c_viz_dir = out_dir / "visualizations_C_overlap"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_all = list_samples(dataset_dir, args.split)
    samples = choose_samples(samples_all, args.sample_ratio, args.max_samples, args.seed)

    print("=" * 80)
    print("SYNTHETIC LABELLED TEST EVALUATION")
    print("=" * 80)
    print(f"Dataset dir : {dataset_dir}")
    print(f"Split       : {args.split}")
    print(f"All samples : {len(samples_all)}")
    print(f"Eval samples: {len(samples)}")
    print(f"Out dir     : {out_dir}")

    model_path = resolve_model_path(results_dir, args.model_path)
    model = load_model_for_inference(model_path)

    rows: List[Dict[str, object]] = []
    numeric_rows: List[Dict[str, float]] = []
    saved_viz = 0

    for start in range(0, len(samples), args.batch_size):
        batch = samples[start:start + args.batch_size]
        x, y, names = load_batch(batch)
        prob = model.predict(x, batch_size=args.batch_size, verbose=0)[..., :3]
        pred_bin = binarize_pred(prob)

        for i, name in enumerate(names):
            m = metrics_one(y[i], pred_bin[i])
            numeric_rows.append(m)
            rows.append({"file": name, **m})

            if saved_viz < args.visualize:
                save_visualization(viz_dir / f"{saved_viz+1:03d}_{Path(name).stem}.png", name, x[i], y[i], prob[i], m)
                save_c_visualization(c_viz_dir / f"{saved_viz+1:03d}_{Path(name).stem}_C.png", name, x[i], y[i], prob[i], m)
                saved_viz += 1

        print(f"[EVAL] {min(start + len(batch), len(samples))}/{len(samples)} done", flush=True)

    summary = weighted_average(numeric_rows)
    summary.update({
        "model_path": str(model_path),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "all_samples_in_split": len(samples_all),
        "evaluated_samples": len(samples),
        "threshold_A": float(THRESHOLDS[0]),
        "threshold_B": float(THRESHOLDS[1]),
        "threshold_C": float(THRESHOLDS[2]),
    })

    metrics_path = out_dir / "metrics.json"
    csv_path = out_dir / "per_image_metrics.csv"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Main segmentation accuracy (mean Dice A/B/C): {summary['accuracy_percent_main_diceABC']:.2f}%")
    print(f"Pixel accuracy A/B/C                    : {summary['pixel_accuracy_percent']:.2f}%")
    print(f"Dice A / B / C                          : {summary['dice_A']:.4f} / {summary['dice_B']:.4f} / {summary['dice_C']:.4f}")
    print(f"IoU  A / B / C                          : {summary['iou_A']:.4f} / {summary['iou_B']:.4f} / {summary['iou_C']:.4f}")
    print(f"Saved metrics                            : {metrics_path}")
    print(f"Saved per-image CSV                      : {csv_path}")
    print(f"Saved visualizations                     : {viz_dir}")
    print(f"Saved C overlap visualizations           : {c_viz_dir}")


if __name__ == "__main__":
    main()
