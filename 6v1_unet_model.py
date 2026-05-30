"""
6v1_unet_model.py
Teacher-Student Hybrid Attention U-Net for NST A/B/C segmentation.

Bản này dùng đúng hướng teacher-student:
- Teacher mạnh hơn Student: base_filters lớn hơn + attention U-Net + edge auxiliary channel.
- Teacher học hard label A/B/C/Edge trước.
- Student học hard label + soft label từ Teacher để giữ shape A/B tốt hơn nhưng vẫn giữ mạnh C.
- Output chính vẫn là A, B, C. Edge chỉ là channel phụ để học biên.
- Resume tự động: tìm checkpoints_teacher/epoch_*.keras và checkpoints_student/epoch_*.keras.
- Batch/Epoch mặc định: 64 / 200.
- Train/val/test cục bộ, xuất JSON + CSV per-image.
- Nếu best test mean_dice_ABC < --min-acc, fine-tune thêm, không train lại từ đầu.

Chạy Colab/project root:
    python 6v1_unet_model.py --epochs 200 --batch-size 64 --min-acc 0.85 --auto-retrain-rounds 1

Lưu ý:
- Nếu T4 OOM ở batch 64, giảm --batch-size còn 16/32. Code vẫn giữ default 64 theo yêu cầu.
- best_for_apply.keras là model tốt nhất để file 7v1 dùng apply vào overlap_raw.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =========================================================
# CONFIG
# =========================================================

ROOT = Path(__file__).resolve().parent
DATASET_DIR_DEFAULT = ROOT / "dataset"
RESULTS_DIR_DEFAULT = ROOT / "results"

IMG_SIZE = 256
IMG_CHANNELS = 1
NUM_MAIN_MASK_CHANNELS = 3          # A, B, C
NUM_OUTPUT_CHANNELS = 4             # A, B, C, edge auxiliary

# C cao để giữ điểm mạnh model cũ ở overlap.
# Edge thêm để học biên A/B giống hướng edge-aware của notebook Mask2Former.
CHANNEL_WEIGHTS = tf.constant([1.25, 1.25, 3.75, 1.00], dtype=tf.float32)
ABC_METRIC_THRESHOLDS = (0.50, 0.50, 0.40)

AUTOTUNE = tf.data.AUTOTUNE
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


@dataclass
class TrainConfig:
    dataset_dir: str
    results_dir: str
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-4
    min_acc: float = 0.85
    auto_retrain_rounds: int = 1
    extra_epochs: int = 50
    patience: int = 15
    force_restart: bool = False
    use_amp: bool = True
    train_teacher: bool = True
    train_student: bool = True
    teacher_base_filters: int = 48
    student_base_filters: int = 24
    hard_loss_weight: float = 0.65
    distill_loss_weight: float = 0.35


def enable_mixed_precision(use_amp: bool) -> None:
    if not use_amp:
        return
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            keras.mixed_precision.set_global_policy("mixed_float16")
            print("[INFO] AMP mixed_float16 enabled.")
        else:
            print("[INFO] No GPU found -> AMP disabled.")
    except Exception as exc:
        print(f"[WARN] Could not enable AMP: {exc}")


# =========================================================
# PATH + DATA VALIDATION
# =========================================================

def ensure_dirs(results_dir: Path) -> Dict[str, Path]:
    paths = {
        "results": results_dir,
        "logs": results_dir / "logs",
        "checkpoints_teacher": results_dir / "checkpoints_teacher",
        "checkpoints_student": results_dir / "checkpoints_student",
        "metric_reports": results_dir / "metric_reports",
        "predicted_masks": results_dir / "predicted_masks",
        "predicted_contours": results_dir / "predicted_contours",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _split_dirs(dataset_dir: Path, split: str) -> Tuple[Path, Path, Path, Path]:
    split_dir = dataset_dir / split
    return (
        split_dir / "images",
        split_dir / "masks_A",
        split_dir / "masks_B",
        split_dir / "masks_C",
    )


def get_file_lists(dataset_dir: Path, split: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    image_dir, mask_a_dir, mask_b_dir, mask_c_dir = _split_dirs(dataset_dir, split)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Missing {image_dir}. Run 3v1 -> 4v1 -> 5v1 first, "
            f"or pass --dataset-dir đúng folder dataset."
        )

    image_paths = sorted(image_dir.glob("*.png"))
    final_image_paths: List[str] = []
    final_mask_a_paths: List[str] = []
    final_mask_b_paths: List[str] = []
    final_mask_c_paths: List[str] = []

    for img_path in image_paths:
        name = img_path.name
        mask_a_path = mask_a_dir / name
        mask_b_path = mask_b_dir / name
        mask_c_path = mask_c_dir / name
        if mask_a_path.exists() and mask_b_path.exists() and mask_c_path.exists():
            final_image_paths.append(str(img_path))
            final_mask_a_paths.append(str(mask_a_path))
            final_mask_b_paths.append(str(mask_b_path))
            final_mask_c_paths.append(str(mask_c_path))
        else:
            print(f"[WARN] Missing mask for {name}, skipped.")

    if len(final_image_paths) == 0:
        raise FileNotFoundError(
            f"No valid samples for split='{split}' in {dataset_dir}. "
            f"Check dataset/{split}/images and masks_A/B/C."
        )

    return final_image_paths, final_mask_a_paths, final_mask_b_paths, final_mask_c_paths


# =========================================================
# TF DATA LOADER
# =========================================================

def read_image(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0..1
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return img


def read_mask(path: tf.Tensor) -> tf.Tensor:
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")
    mask = tf.cast(mask > 127, tf.float32)
    mask.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return mask


def make_boundary_channel(mask_abc: tf.Tensor, k: int = 3) -> tf.Tensor:
    """
    Tạo edge/boundary target từ union(A,B,C). Edge là channel phụ giúp học ranh giới yếu.
    mask_abc: [H, W, 3], value 0/1.
    return: [H, W, 1]
    """
    fg = tf.reduce_max(mask_abc, axis=-1, keepdims=True)
    x = tf.expand_dims(fg, axis=0)  # [1,H,W,1]
    dil = tf.nn.max_pool2d(x, ksize=k, strides=1, padding="SAME")
    ero = 1.0 - tf.nn.max_pool2d(1.0 - x, ksize=k, strides=1, padding="SAME")
    edge = tf.clip_by_value(dil - ero, 0.0, 1.0)
    edge = tf.squeeze(edge, axis=0)
    edge.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return edge


def load_sample(image_path: tf.Tensor, mask_a_path: tf.Tensor, mask_b_path: tf.Tensor, mask_c_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = read_image(image_path)
    mask_a = read_mask(mask_a_path)
    mask_b = read_mask(mask_b_path)
    mask_c = read_mask(mask_c_path)
    mask_abc = tf.concat([mask_a, mask_b, mask_c], axis=-1)
    edge = make_boundary_channel(mask_abc)
    mask_abce = tf.concat([mask_abc, edge], axis=-1)
    mask_abce.set_shape([IMG_SIZE, IMG_SIZE, NUM_OUTPUT_CHANNELS])
    return image, mask_abce


def augment_sample(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Augment nhẹ, không crop/resize méo hình NST."""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # Noise/contrast nhẹ để giảm fail ca mờ, không phá nhãn.
    if tf.random.uniform(()) > 0.65:
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    if tf.random.uniform(()) > 0.70:
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.015, dtype=tf.float32)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image, mask


def make_dataset(dataset_dir: Path, split: str, batch_size: int, shuffle: bool = False, augment: bool = False) -> Tuple[tf.data.Dataset, int]:
    image_paths, mask_a_paths, mask_b_paths, mask_c_paths = get_file_lists(dataset_dir, split)
    n = len(image_paths)
    print(f"[DATA] {split}: {n} samples")

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_a_paths, mask_b_paths, mask_c_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(n, 4096), reshuffle_each_iteration=True, seed=SEED)
    ds = ds.map(load_sample, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(augment_sample, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds, n


def make_distill_dataset(ds: tf.data.Dataset, teacher: keras.Model) -> tf.data.Dataset:
    """
    Dataset cho Student: y_true = concat(hard_abce, teacher_soft_abce).
    Teacher frozen, chạy trên batch nên không cần lưu soft label ra disk.
    """
    teacher.trainable = False

    def add_soft(x: tf.Tensor, y_hard: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        y_soft = tf.stop_gradient(teacher(x, training=False))
        y = tf.concat([y_hard, y_soft], axis=-1)
        y.set_shape([None, IMG_SIZE, IMG_SIZE, NUM_OUTPUT_CHANNELS * 2])
        return x, y

    return ds.map(add_soft, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


# =========================================================
# MODEL: ATTENTION U-NET + EDGE AUXILIARY OUTPUT
# =========================================================

def conv_block(x: tf.Tensor, filters: int, dropout: float = 0.0) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout > 0:
        x = layers.SpatialDropout2D(dropout)(x)
    return x


def encoder_block(x: tf.Tensor, filters: int, dropout: float = 0.0) -> Tuple[tf.Tensor, tf.Tensor]:
    skip = conv_block(x, filters, dropout=dropout)
    pooled = layers.MaxPooling2D(pool_size=(2, 2))(skip)
    return skip, pooled


def attention_gate(skip: tf.Tensor, gating: tf.Tensor, filters: int) -> tf.Tensor:
    theta = layers.Conv2D(filters, 1, padding="same", use_bias=False)(skip)
    phi = layers.Conv2D(filters, 1, padding="same", use_bias=False)(gating)
    act = layers.Activation("relu")(layers.Add()([theta, phi]))
    psi = layers.Conv2D(1, 1, padding="same")(act)
    psi = layers.Activation("sigmoid")(psi)
    return layers.Multiply()([skip, psi])


def decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int, dropout: float = 0.0) -> tf.Tensor:
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    skip = attention_gate(skip, x, filters)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters, dropout=dropout)
    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS),
    output_channels: int = NUM_OUTPUT_CHANNELS,
    base_filters: int = 32,
    name: str = "Hybrid_Attention_UNet_ABCE",
) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="image")
    f = base_filters

    s1, p1 = encoder_block(inputs, f, dropout=0.00)
    s2, p2 = encoder_block(p1, f * 2, dropout=0.00)
    s3, p3 = encoder_block(p2, f * 4, dropout=0.05)
    s4, p4 = encoder_block(p3, f * 8, dropout=0.10)

    b = conv_block(p4, f * 16, dropout=0.15)

    d1 = decoder_block(b, s4, f * 8, dropout=0.10)
    d2 = decoder_block(d1, s3, f * 4, dropout=0.05)
    d3 = decoder_block(d2, s2, f * 2, dropout=0.00)
    d4 = decoder_block(d3, s1, f, dropout=0.00)

    outputs = layers.Conv2D(
        output_channels, 1, padding="same", activation="sigmoid", dtype="float32", name="mask_abce"
    )(d4)
    return keras.Model(inputs, outputs, name=name)


# =========================================================
# LOSS + METRICS
# =========================================================

def weighted_bce(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true[..., :NUM_OUTPUT_CHANNELS], tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * CHANNEL_WEIGHTS)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true = tf.cast(y_true[..., :NUM_OUTPUT_CHANNELS], tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    weighted = dice * CHANNEL_WEIGHTS
    weighted = tf.reduce_sum(weighted, axis=-1) / tf.reduce_sum(CHANNEL_WEIGHTS)
    return 1.0 - tf.reduce_mean(weighted)


def hybrid_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return weighted_bce(y_true, y_pred) + dice_loss(y_true, y_pred)


def student_distill_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    y_true 8 channels: [hard_A,B,C,Edge, soft_A,B,C,Edge].
    Student vừa học label thật vừa học soft output của Teacher.
    """
    hard = y_true[..., :NUM_OUTPUT_CHANNELS]
    soft = y_true[..., NUM_OUTPUT_CHANNELS:NUM_OUTPUT_CHANNELS * 2]
    hard_loss = hybrid_loss(hard, y_pred)

    # Distill trên probability, weighted để C vẫn quan trọng.
    mse = tf.square(tf.cast(soft, tf.float32) - tf.cast(y_pred, tf.float32))
    distill = tf.reduce_mean(mse * CHANNEL_WEIGHTS)
    return CURRENT_HARD_LOSS_WEIGHT * hard_loss + CURRENT_DISTILL_LOSS_WEIGHT * distill


# Giá trị này được set lại trong compile_student(). Dùng biến global để Keras serialize đơn giản.
CURRENT_HARD_LOSS_WEIGHT = tf.Variable(0.65, trainable=False, dtype=tf.float32)
CURRENT_DISTILL_LOSS_WEIGHT = tf.Variable(0.35, trainable=False, dtype=tf.float32)


def _threshold_for_channel(index: int) -> float:
    if index == 2:
        return ABC_METRIC_THRESHOLDS[2]
    return ABC_METRIC_THRESHOLDS[index] if index < 3 else 0.50


def dice_channel(index: int, name: str):
    def metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = y_true[..., :NUM_OUTPUT_CHANNELS]
        smooth = 1e-6
        yt = tf.cast(y_true[..., index], tf.float32)
        yp = tf.cast(y_pred[..., index] > _threshold_for_channel(index), tf.float32)
        intersection = tf.reduce_sum(yt * yp, axis=[1, 2])
        denominator = tf.reduce_sum(yt + yp, axis=[1, 2])
        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        return tf.reduce_mean(dice)
    metric.__name__ = name
    return metric


def iou_channel(index: int, name: str):
    def metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = y_true[..., :NUM_OUTPUT_CHANNELS]
        smooth = 1e-6
        yt = tf.cast(y_true[..., index], tf.float32)
        yp = tf.cast(y_pred[..., index] > _threshold_for_channel(index), tf.float32)
        intersection = tf.reduce_sum(yt * yp, axis=[1, 2])
        union = tf.reduce_sum(yt + yp, axis=[1, 2]) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return tf.reduce_mean(iou)
    metric.__name__ = name
    return metric


def mean_dice_abc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    vals = [dice_channel(i, f"dice_{i}")(y_true, y_pred) for i in range(3)]
    return tf.reduce_mean(tf.stack(vals))


def pixel_acc_abc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = y_true[..., :NUM_OUTPUT_CHANNELS]
    yt = tf.cast(y_true[..., :3] > 0.5, tf.bool)
    thresholds = tf.constant(ABC_METRIC_THRESHOLDS, dtype=tf.float32)
    yp = tf.cast(y_pred[..., :3] > thresholds, tf.bool)
    return tf.reduce_mean(tf.cast(tf.equal(yt, yp), tf.float32))


METRICS = [
    dice_channel(0, "dice_A"),
    dice_channel(1, "dice_B"),
    dice_channel(2, "dice_C"),
    dice_channel(3, "dice_edge"),
    iou_channel(0, "iou_A"),
    iou_channel(1, "iou_B"),
    iou_channel(2, "iou_C"),
    mean_dice_abc,
    pixel_acc_abc,
]

CUSTOM_OBJECTS = {
    "hybrid_loss": hybrid_loss,
    "student_distill_loss": student_distill_loss,
    "weighted_bce": weighted_bce,
    "dice_loss": dice_loss,
    "mean_dice_abc": mean_dice_abc,
    "pixel_acc_abc": pixel_acc_abc,
    "dice_A": dice_channel(0, "dice_A"),
    "dice_B": dice_channel(1, "dice_B"),
    "dice_C": dice_channel(2, "dice_C"),
    "dice_edge": dice_channel(3, "dice_edge"),
    "iou_A": iou_channel(0, "iou_A"),
    "iou_B": iou_channel(1, "iou_B"),
    "iou_C": iou_channel(2, "iou_C"),
}


def compile_teacher(model: keras.Model, lr: float) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=hybrid_loss, metrics=METRICS)
    return model


def compile_student(model: keras.Model, lr: float, hard_weight: float, distill_weight: float) -> keras.Model:
    keras.backend.set_value(CURRENT_HARD_LOSS_WEIGHT, hard_weight)
    keras.backend.set_value(CURRENT_DISTILL_LOSS_WEIGHT, distill_weight)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=student_distill_loss, metrics=METRICS)
    return model


# =========================================================
# CHECKPOINT RESUME
# =========================================================

def parse_epoch_from_path(path: Path) -> int:
    m = re.search(r"epoch_(\d+)\.keras$", path.name)
    return int(m.group(1)) if m else 0


def find_latest_epoch_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    candidates = sorted(checkpoint_dir.glob("epoch_*.keras"), key=parse_epoch_from_path)
    return candidates[-1] if candidates else None


def safe_load_model(path: Path, compile_fn=None) -> keras.Model:
    print(f"[RESUME] Loading checkpoint: {path}")
    try:
        model = keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=True)
        print("[RESUME] Loaded with optimizer state.")
        return model
    except Exception as exc:
        print(f"[WARN] load_model compile=True failed: {exc}")
        print("[RESUME] Loading compile=False, then recompile.")
        model = keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
        if compile_fn is not None:
            model = compile_fn(model)
        return model


def build_or_resume_role(
    results_dir: Path,
    role: str,
    lr: float,
    force_restart: bool,
    base_filters: int,
    compile_fn,
) -> Tuple[keras.Model, int]:
    ckpt_dir = results_dir / f"checkpoints_{role}"
    best_ckpt = results_dir / f"best_{role}.keras"
    latest_ckpt = find_latest_epoch_checkpoint(ckpt_dir)

    if not force_restart and latest_ckpt is not None:
        initial_epoch = parse_epoch_from_path(latest_ckpt)
        model = safe_load_model(latest_ckpt, compile_fn=compile_fn)
        print(f"[RESUME][{role}] Continue from epoch {initial_epoch + 1}.")
        return model, initial_epoch

    if not force_restart and best_ckpt.exists():
        model = safe_load_model(best_ckpt, compile_fn=compile_fn)
        print(f"[RESUME][{role}] Found best_{role}.keras but no epoch checkpoint -> continue from epoch 1.")
        return model, 0

    print(f"[MODEL] Starting new {role.upper()} model. base_filters={base_filters}")
    model = build_unet(base_filters=base_filters, name=f"{role.capitalize()}_Hybrid_Attention_UNet_ABCE")
    model = compile_fn(model)
    return model, 0


# =========================================================
# EVALUATION REPORT
# =========================================================

def _np_metrics(y_true_abc: np.ndarray, y_prob_abc: np.ndarray) -> Dict[str, float]:
    thresholds = np.array(ABC_METRIC_THRESHOLDS, dtype=np.float32).reshape(1, 1, 1, 3)
    y_true = y_true_abc > 0.5
    y_pred = y_prob_abc > thresholds

    per: Dict[str, float] = {}
    dice_values = []
    iou_values = []
    names = ["A", "B", "C"]

    for i, name in enumerate(names):
        yt = y_true[..., i]
        yp = y_pred[..., i]
        inter = np.logical_and(yt, yp).sum(axis=(1, 2)).astype(np.float64)
        denom = yt.sum(axis=(1, 2)) + yp.sum(axis=(1, 2))
        union = np.logical_or(yt, yp).sum(axis=(1, 2)).astype(np.float64)
        dice = np.where(denom == 0, 1.0, (2.0 * inter) / np.maximum(denom, 1))
        iou = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
        per[f"dice_{name}"] = float(np.mean(dice))
        per[f"iou_{name}"] = float(np.mean(iou))
        dice_values.append(per[f"dice_{name}"])
        iou_values.append(per[f"iou_{name}"])

    per["mean_dice_ABC"] = float(np.mean(dice_values))
    per["mean_iou_ABC"] = float(np.mean(iou_values))
    per["pixel_acc_ABC"] = float(np.mean(y_true == y_pred))
    return per


def evaluate_split(
    model: keras.Model,
    dataset_dir: Path,
    split: str,
    batch_size: int,
    report_dir: Path,
    model_name: str,
) -> Dict[str, float]:
    image_paths, mask_a_paths, mask_b_paths, mask_c_paths = get_file_lists(dataset_dir, split)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float | str]] = []
    agg_chunks: List[Dict[str, float]] = []

    for start in range(0, len(image_paths), batch_size):
        batch_imgs = []
        batch_true = []
        batch_names = []
        for img_path, ma_path, mb_path, mc_path in zip(
            image_paths[start:start + batch_size],
            mask_a_paths[start:start + batch_size],
            mask_b_paths[start:start + batch_size],
            mask_c_paths[start:start + batch_size],
        ):
            img = tf.image.decode_png(tf.io.read_file(img_path), channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
            batch_imgs.append(img.numpy())

            masks = []
            for p in (ma_path, mb_path, mc_path):
                m = tf.image.decode_png(tf.io.read_file(p), channels=1)
                m = tf.image.resize(m, (IMG_SIZE, IMG_SIZE), method="nearest")
                masks.append((m.numpy() > 127).astype(np.float32))
            batch_true.append(np.concatenate(masks, axis=-1))
            batch_names.append(Path(img_path).name)

        x = np.stack(batch_imgs, axis=0).astype(np.float32)
        y = np.stack(batch_true, axis=0).astype(np.float32)
        pred = model.predict(x, batch_size=batch_size, verbose=0)[..., :3]
        chunk_metrics = _np_metrics(y, pred)
        agg_chunks.append({**chunk_metrics, "n": len(batch_names)})

        for i, name in enumerate(batch_names):
            m = _np_metrics(y[i:i + 1], pred[i:i + 1])
            rows.append({"file": name, **m})

    total_n = sum(int(c["n"]) for c in agg_chunks)
    metrics: Dict[str, float] = {}
    for key in agg_chunks[0].keys():
        if key == "n":
            continue
        metrics[key] = float(sum(c[key] * c["n"] for c in agg_chunks) / max(1, total_n))
    metrics["samples"] = float(total_n)
    metrics["accuracy_percent_main"] = metrics["mean_dice_ABC"] * 100.0
    metrics["pixel_accuracy_percent"] = metrics["pixel_acc_ABC"] * 100.0

    json_path = report_dir / f"{model_name}_{split}_metrics.json"
    csv_path = report_dir / f"{model_name}_{split}_per_image.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"[EVAL][{model_name}] {split}:")
    print(json.dumps(metrics, indent=2))
    print(f"[EVAL] Saved {json_path}")
    return metrics


def eval_all_splits(model: keras.Model, dataset_dir: Path, batch_size: int, report_dir: Path, model_name: str) -> Dict[str, Dict[str, float]]:
    out = {}
    for split in ["train", "val", "test"]:
        out[split] = evaluate_split(model, dataset_dir, split, batch_size, report_dir, model_name=model_name)
    return out


# =========================================================
# TRAIN HELPERS
# =========================================================

def make_callbacks(results_dir: Path, role: str, patience: int, append_log: bool) -> List[keras.callbacks.Callback]:
    checkpoint_dir = results_dir / f"checkpoints_{role}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(results_dir / f"best_{role}.keras"),
            monitor="val_mean_dice_abc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "epoch_{epoch:03d}.keras"),
            save_best_only=False,
            save_freq="epoch",
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mean_dice_abc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mean_dice_abc",
            mode="max",
            factor=0.5,
            patience=max(3, patience // 4),
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=str(results_dir / "logs" / f"{role}_train_log.csv"),
            append=append_log,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def train_role(
    model: keras.Model,
    role: str,
    initial_epoch: int,
    target_epochs: int,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    results_dir: Path,
    patience: int,
) -> keras.Model:
    if target_epochs <= initial_epoch:
        print(f"[TRAIN][{role}] target_epochs={target_epochs} <= initial_epoch={initial_epoch}, skip fit.")
        return model

    print(f"[TRAIN][{role}] Fit from epoch {initial_epoch + 1} to {target_epochs}")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=target_epochs,
        initial_epoch=initial_epoch,
        callbacks=make_callbacks(results_dir, role=role, patience=patience, append_log=initial_epoch > 0),
        verbose=1,
    )
    final_path = results_dir / f"final_{role}.keras"
    model.save(final_path)
    print(f"[SAVE][{role}] Final model saved to {final_path}")
    return model


def load_best_if_exists(results_dir: Path, role: str, compile_fn) -> keras.Model:
    best_path = results_dir / f"best_{role}.keras"
    final_path = results_dir / f"final_{role}.keras"
    if best_path.exists():
        return safe_load_model(best_path, compile_fn=compile_fn)
    if final_path.exists():
        return safe_load_model(final_path, compile_fn=compile_fn)
    raise FileNotFoundError(f"Missing both {best_path} and {final_path}")


def choose_best_for_apply(results_dir: Path, all_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    best_role = max(all_metrics.keys(), key=lambda r: all_metrics[r]["test"]["mean_dice_ABC"])
    src = results_dir / f"best_{best_role}.keras"
    if not src.exists():
        src = results_dir / f"final_{best_role}.keras"
    dst = results_dir / "best_for_apply.keras"
    shutil.copy2(src, dst)

    # Alias để script 7v1 cũ hoặc notebook dễ load.
    for alias in ["best_unet.keras", "best_hybrid_unet.keras"]:
        try:
            shutil.copy2(src, results_dir / alias)
        except Exception as exc:
            print(f"[WARN] Could not create alias {alias}: {exc}")

    summary_path = results_dir / "best_model_summary.json"
    summary = {
        "best_role": best_role,
        "source_model": str(src),
        "best_for_apply": str(dst),
        "test_mean_dice_ABC": all_metrics[best_role]["test"]["mean_dice_ABC"],
        "test_accuracy_percent_main": all_metrics[best_role]["test"]["accuracy_percent_main"],
        "all_metrics": all_metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[BEST] {best_role} chosen for apply. Saved: {dst}")
    print(f"[BEST] Summary: {summary_path}")
    return best_role


# =========================================================
# MAIN
# =========================================================

def main(args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR_DEFAULT))
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR_DEFAULT))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-acc", type=float, default=0.85, help="Minimum mean_dice_ABC required on test set.")
    parser.add_argument("--auto-retrain-rounds", type=int, default=1, help="Fine-tune thêm nếu test acc < min-acc.")
    parser.add_argument("--extra-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-student", action="store_true")
    parser.add_argument("--teacher-base-filters", type=int, default=48)
    parser.add_argument("--student-base-filters", type=int, default=24)
    parser.add_argument("--hard-loss-weight", type=float, default=0.65)
    parser.add_argument("--distill-loss-weight", type=float, default=0.35)
    ns = parser.parse_args(args)

    cfg = TrainConfig(
        dataset_dir=str(Path(ns.dataset_dir).resolve()),
        results_dir=str(Path(ns.results_dir).resolve()),
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        learning_rate=ns.learning_rate,
        min_acc=ns.min_acc,
        auto_retrain_rounds=ns.auto_retrain_rounds,
        extra_epochs=ns.extra_epochs,
        patience=ns.patience,
        force_restart=ns.force_restart,
        use_amp=not ns.no_amp,
        train_teacher=not ns.skip_teacher,
        train_student=not ns.skip_student,
        teacher_base_filters=ns.teacher_base_filters,
        student_base_filters=ns.student_base_filters,
        hard_loss_weight=ns.hard_loss_weight,
        distill_loss_weight=ns.distill_loss_weight,
    )

    dataset_dir = Path(cfg.dataset_dir)
    results_dir = Path(cfg.results_dir)
    paths = ensure_dirs(results_dir)
    with open(results_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    enable_mixed_precision(cfg.use_amp)

    train_ds, _ = make_dataset(dataset_dir, "train", cfg.batch_size, shuffle=True, augment=True)
    val_ds, _ = make_dataset(dataset_dir, "val", cfg.batch_size, shuffle=False, augment=False)
    _ = get_file_lists(dataset_dir, "test")

    teacher_compile = lambda m: compile_teacher(m, cfg.learning_rate)
    student_compile = lambda m: compile_student(m, cfg.learning_rate, cfg.hard_loss_weight, cfg.distill_loss_weight)

    teacher_model: Optional[keras.Model] = None
    student_model: Optional[keras.Model] = None
    teacher_initial_epoch = 0
    student_initial_epoch = 0

    if cfg.train_teacher:
        teacher_model, teacher_initial_epoch = build_or_resume_role(
            results_dir=results_dir,
            role="teacher",
            lr=cfg.learning_rate,
            force_restart=cfg.force_restart,
            base_filters=cfg.teacher_base_filters,
            compile_fn=teacher_compile,
        )
        teacher_model.summary()
    else:
        teacher_model = load_best_if_exists(results_dir, "teacher", teacher_compile)

    if cfg.train_student:
        student_model, student_initial_epoch = build_or_resume_role(
            results_dir=results_dir,
            role="student",
            lr=cfg.learning_rate,
            force_restart=cfg.force_restart,
            base_filters=cfg.student_base_filters,
            compile_fn=student_compile,
        )
        student_model.summary()

    target_epochs = cfg.epochs
    best_role = "teacher"
    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for retrain_round in range(cfg.auto_retrain_rounds + 1):
        print(f"\n========== ROUND {retrain_round + 1}/{cfg.auto_retrain_rounds + 1} ==========")

        if cfg.train_teacher and teacher_model is not None:
            teacher_model = train_role(
                model=teacher_model,
                role="teacher",
                initial_epoch=teacher_initial_epoch,
                target_epochs=target_epochs,
                train_ds=train_ds,
                val_ds=val_ds,
                results_dir=results_dir,
                patience=cfg.patience,
            )
            teacher_model = load_best_if_exists(results_dir, "teacher", teacher_compile)
        elif teacher_model is not None:
            print("[TRAIN][teacher] skipped, using loaded teacher.")

        if teacher_model is None:
            raise RuntimeError("Teacher model is required for student distillation.")

        # Student học hard label + soft output của teacher.
        if cfg.train_student and student_model is not None:
            train_ds_student = make_distill_dataset(train_ds, teacher_model)
            val_ds_student = make_distill_dataset(val_ds, teacher_model)
            student_model = train_role(
                model=student_model,
                role="student",
                initial_epoch=student_initial_epoch,
                target_epochs=target_epochs,
                train_ds=train_ds_student,
                val_ds=val_ds_student,
                results_dir=results_dir,
                patience=cfg.patience,
            )
            student_model = load_best_if_exists(results_dir, "student", student_compile)

        all_metrics = {}
        if teacher_model is not None:
            all_metrics["teacher"] = eval_all_splits(
                teacher_model, dataset_dir, cfg.batch_size, paths["metric_reports"], model_name="teacher"
            )
        if student_model is not None:
            all_metrics["student"] = eval_all_splits(
                student_model, dataset_dir, cfg.batch_size, paths["metric_reports"], model_name="student"
            )

        best_role = choose_best_for_apply(results_dir, all_metrics)
        best_acc = all_metrics[best_role]["test"]["mean_dice_ABC"]
        print(f"[CHECK] Best={best_role} test mean_dice_ABC={best_acc:.4f} ({best_acc*100:.2f}%) | target={cfg.min_acc:.2f}")

        if best_acc >= cfg.min_acc:
            print("[OK] Accuracy đạt ngưỡng. Dừng train.")
            break

        if retrain_round >= cfg.auto_retrain_rounds:
            print("[WARN] Accuracy vẫn chưa đạt ngưỡng sau số vòng retrain cho phép.")
            break

        print(f"[RETRAIN] acc < {cfg.min_acc:.2f}. Fine-tune thêm {cfg.extra_epochs} epochs.")
        teacher_initial_epoch = target_epochs
        student_initial_epoch = target_epochs
        target_epochs += cfg.extra_epochs

        # Giảm LR nhẹ cho fine-tune, không reset weight.
        for role, model in [("teacher", teacher_model), ("student", student_model)]:
            if model is None:
                continue
            try:
                old_lr = float(keras.backend.get_value(model.optimizer.learning_rate))
                new_lr = max(old_lr * 0.5, 1e-6)
                keras.backend.set_value(model.optimizer.learning_rate, new_lr)
                print(f"[RETRAIN][{role}] LR: {old_lr:g} -> {new_lr:g}")
            except Exception:
                pass

    print("[DONE] Teacher-student train/eval completed.")


if __name__ == "__main__":
    main()
