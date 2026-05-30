"""
7v1_predict_real_overlap.py
Apply model vào ảnh overlap_raw, xuất mask/overlay và tách thành 2 file NST A/B theo ảnh gốc.

Điểm mới:
- Path lấy theo vị trí file script, không phụ thuộc đang đứng ở folder nào.
- Ưu tiên load bản predict nhẹ results/best_for_apply_inference.keras hoặc best_for_apply.keras, rồi mới tới best_student/teacher.
- Model predict ở 256 padding, sau đó map mask về kích thước ảnh gốc.
- Sau khi label A/B/C, tạo 2 file cùng thư mục separated_chromosomes:
    <ten_anh_goc>_A.png
    <ten_anh_goc>_B.png
- Vùng C được OR vào cả A và B để giữ full hình thái NST.
- Vùng bị đè/overlap C được fill lại bằng inpainting từ phần visible của chính NST đó.

Chạy:
    python 7v1_predict_real_overlap.py
    python 7v1_predict_real_overlap.py --input-dir source_data/overlap_raw --output-dir results/real_predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras


# =========================
# CONFIG
# =========================

ROOT = Path(__file__).resolve().parent
INPUT_DIR_DEFAULT = ROOT / "source_data" / "overlap_raw"
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR_DEFAULT = RESULTS_DIR / "real_predictions"
IMG_SIZE = 256

THRESH_A = 0.50
THRESH_B = 0.50
THRESH_C = 0.40

MODEL_CANDIDATES = [
    RESULTS_DIR / "best_for_apply_inference.keras",
    RESULTS_DIR / "best_for_apply.keras",
    RESULTS_DIR / "best_student_inference.keras",
    RESULTS_DIR / "best_teacher_inference.keras",
    RESULTS_DIR / "best_student.keras",
    RESULTS_DIR / "best_teacher.keras",
    RESULTS_DIR / "best_hybrid_unet.keras",
    RESULTS_DIR / "best_unet.keras",
    RESULTS_DIR / "final_student.keras",
    RESULTS_DIR / "final_teacher.keras",
    RESULTS_DIR / "final_unet.keras",
]


# =========================
# PREPROCESS + RESTORE SIZE
# =========================

def resize_with_padding_image(img: Image.Image, target_size: int = 256) -> Tuple[Image.Image, Dict[str, int]]:
    """Convert grayscale, resize giữ tỉ lệ, padding nền trắng."""
    img = img.convert("L")
    w, h = img.size
    scale = min(target_size / w, target_size / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("L", (target_size, target_size), 255)

    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))

    meta = {
        "orig_w": w,
        "orig_h": h,
        "new_w": new_w,
        "new_h": new_h,
        "paste_x": paste_x,
        "paste_y": paste_y,
    }
    return canvas, meta


def prepare_input(img: Image.Image) -> np.ndarray:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def restore_mask_to_original(mask_256: np.ndarray, meta: Dict[str, int]) -> np.ndarray:
    """Unpad mask 256 rồi resize nearest về kích thước ảnh gốc."""
    x = meta["paste_x"]
    y = meta["paste_y"]
    nw = meta["new_w"]
    nh = meta["new_h"]
    ow = meta["orig_w"]
    oh = meta["orig_h"]

    crop = mask_256[y:y + nh, x:x + nw].astype(np.uint8) * 255
    restored = Image.fromarray(crop, mode="L").resize((ow, oh), Image.NEAREST)
    return np.array(restored) > 127


# =========================
# POSTPROCESS
# =========================

def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(path)


def keep_largest_components(mask: np.ndarray, keep: int = 1) -> np.ndarray:
    """Giữ component lớn nhất để giảm noise; keep=0 thì bỏ qua."""
    if keep <= 0:
        return mask.astype(bool)
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1][:keep] + 1
    return np.isin(labels, order)


def clean_mask(mask: np.ndarray, keep_components: int = 1) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = keep_largest_components(cleaned > 0, keep=keep_components)
    return cleaned.astype(bool)


def mask_to_contour(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(mask_uint8)
    cv2.drawContours(contour_img, contours, -1, 255, thickness=1)
    return contour_img


def make_overlay(base_img: Image.Image, mask_A: np.ndarray, mask_B: np.ndarray, mask_C: np.ndarray) -> Image.Image:
    base = np.array(base_img.convert("RGB")).astype(np.float32)
    overlay = base.copy()

    A = mask_A.astype(bool)
    B = mask_B.astype(bool)
    C = mask_C.astype(bool)

    overlay[A] = overlay[A] * 0.4 + np.array([255, 0, 0]) * 0.6
    overlay[B] = overlay[B] * 0.4 + np.array([0, 255, 0]) * 0.6
    overlay[C] = overlay[C] * 0.3 + np.array([255, 255, 0]) * 0.7

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def make_contour_preview(base_img: Image.Image, mask_A: np.ndarray, mask_B: np.ndarray, mask_C: np.ndarray) -> Image.Image:
    base = np.array(base_img.convert("RGB")).copy()

    contour_A = mask_to_contour(mask_A) > 0
    contour_B = mask_to_contour(mask_B) > 0
    contour_C = mask_to_contour(mask_C) > 0

    base[contour_A] = [255, 0, 0]
    base[contour_B] = [0, 255, 0]
    base[contour_C] = [255, 255, 0]

    return Image.fromarray(base)


def to_rgb(img) -> Image.Image:
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(img.astype(np.uint8))
    return img.convert("RGB")


def add_title(img: Image.Image, title: str, bar_height: int = 28) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_height), (255, 255, 255))
    canvas.paste(img, (0, bar_height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w, bar_height], fill=(230, 230, 230))
    draw.text((8, 6), title, fill=(0, 0, 0))
    return canvas


def make_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    box = 16
    gap = 8
    line_h = 24
    legend_items = [
        ((255, 0, 0), "A = Chromosome A"),
        ((0, 255, 0), "B = Chromosome B"),
        ((255, 255, 0), "C = Overlap"),
    ]
    draw.rectangle([x - 8, y - 8, x + 220, y + 78], fill=(255, 255, 255), outline=(0, 0, 0))
    for i, (color, text) in enumerate(legend_items):
        yy = y + i * line_h
        draw.rectangle([x, yy, x + box, yy + box], fill=color, outline=(0, 0, 0))
        draw.text((x + box + gap, yy), text, fill=(0, 0, 0))


def make_visualization(base_img: Image.Image, overlay_img: Image.Image, contour_img: Image.Image, mask_A: np.ndarray, mask_B: np.ndarray, mask_C: np.ndarray, save_path: Path) -> None:
    mask_A_img = Image.fromarray((mask_A.astype(np.uint8) * 255), mode="L")
    mask_B_img = Image.fromarray((mask_B.astype(np.uint8) * 255), mode="L")
    mask_C_img = Image.fromarray((mask_C.astype(np.uint8) * 255), mode="L")

    panels = [
        add_title(to_rgb(base_img), "Input"),
        add_title(to_rgb(overlay_img), "Overlay"),
        add_title(to_rgb(contour_img), "Contour"),
        add_title(to_rgb(mask_A_img), "Mask A"),
        add_title(to_rgb(mask_B_img), "Mask B"),
        add_title(to_rgb(mask_C_img), "Mask C"),
    ]

    panel_w, panel_h = panels[0].size
    cols = 3
    rows = 2
    margin = 10
    legend_h = 100

    canvas_w = cols * panel_w + (cols + 1) * margin
    canvas_h = rows * panel_h + (rows + 1) * margin + legend_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))

    idx = 0
    for r in range(rows):
        for c in range(cols):
            x = margin + c * (panel_w + margin)
            y = margin + r * (panel_h + margin)
            canvas.paste(panels[idx], (x, y))
            idx += 1

    draw = ImageDraw.Draw(canvas)
    make_legend(draw, 20, rows * panel_h + (rows + 1) * margin + 10)
    canvas.save(save_path)


# =========================
# SEPARATE A/B + FILL OCCLUDED AREA
# =========================

def inpaint_instance(gray: np.ndarray, instance_mask: np.ndarray, overlap_mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Fill vùng overlap C cho từng NST bằng inpainting.
    Để tránh lấy texture của NST khác, vùng ngoài instance được set trắng trước khi inpaint.
    """
    gray_u8 = gray.astype(np.uint8)
    work = gray_u8.copy()
    work[~instance_mask] = 255

    fill_region = (instance_mask & overlap_mask).astype(np.uint8) * 255
    if fill_region.sum() == 0:
        return work

    # Nới vùng fill rất nhẹ để tránh viền đè còn sót.
    kernel = np.ones((3, 3), np.uint8)
    fill_region = cv2.dilate(fill_region, kernel, iterations=1)
    fill_region[~instance_mask] = 0

    filled = cv2.inpaint(work, fill_region, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return filled


def rgba_from_instance(gray: np.ndarray, instance_mask: np.ndarray, overlap_mask: np.ndarray, transparent: bool = True) -> Image.Image:
    filled = inpaint_instance(gray, instance_mask, overlap_mask)
    if transparent:
        rgba = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
        rgba[..., 0] = filled
        rgba[..., 1] = filled
        rgba[..., 2] = filled
        rgba[..., 3] = instance_mask.astype(np.uint8) * 255
        return Image.fromarray(rgba, mode="RGBA")

    out = np.full_like(gray, 255, dtype=np.uint8)
    out[instance_mask] = filled[instance_mask]
    return Image.fromarray(out, mode="L")


def save_separated_ab(
    original_gray: Image.Image,
    stem: str,
    mask_A: np.ndarray,
    mask_B: np.ndarray,
    mask_C: np.ndarray,
    out_dir: Path,
    transparent: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gray = np.array(original_gray.convert("L"))

    # C là vùng overlap, nên để giữ amodal shape thì C thuộc cả A và B.
    A_full = clean_mask(mask_A | mask_C, keep_components=1)
    B_full = clean_mask(mask_B | mask_C, keep_components=1)
    C_clean = clean_mask(mask_C, keep_components=1)

    img_A = rgba_from_instance(gray, A_full, C_clean, transparent=transparent)
    img_B = rgba_from_instance(gray, B_full, C_clean, transparent=transparent)

    img_A.save(out_dir / f"{stem}_A.png")
    img_B.save(out_dir / f"{stem}_B.png")
    save_binary_mask(A_full, out_dir / f"{stem}_A_mask.png")
    save_binary_mask(B_full, out_dir / f"{stem}_B_mask.png")


# =========================
# IO + MAIN
# =========================

def get_image_paths(input_dir: Path) -> List[Path]:
    patterns = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    image_paths: List[Path] = []
    for pattern in patterns:
        image_paths.extend(input_dir.glob(pattern))
    return sorted(image_paths)


def find_model_path(model_path_arg: str | None) -> Path:
    if model_path_arg:
        p = Path(model_path_arg)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Model path not found: {p}")
        return p

    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No model found. Expected one of: " + ", ".join(str(p) for p in MODEL_CANDIDATES)
    )


def make_output_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "processed_images": output_dir / "processed_images_256",
        "masks_A": output_dir / "masks_A_original_size",
        "masks_B": output_dir / "masks_B_original_size",
        "masks_C": output_dir / "masks_C_original_size",
        "contours": output_dir / "contours_original_size",
        "overlays": output_dir / "overlays_original_size",
        "visualizations": output_dir / "visualizations_256",
        "separated": output_dir / "separated_chromosomes",
    }
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR_DEFAULT))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR_DEFAULT))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--thresh-a", type=float, default=THRESH_A)
    parser.add_argument("--thresh-b", type=float, default=THRESH_B)
    parser.add_argument("--thresh-c", type=float, default=THRESH_C)
    parser.add_argument("--white-background", action="store_true", help="Nếu bật, A/B output nền trắng thay vì transparent PNG.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    model_path = find_model_path(args.model_path)
    out = make_output_dirs(output_dir)
    image_paths = get_image_paths(input_dir)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No image files found in {input_dir}. Supported: png, jpg, jpeg.")

    print(f"Found {len(image_paths)} real overlap images.")
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    for idx, img_path in enumerate(image_paths, start=1):
        stem = img_path.stem
        name = stem + ".png"

        original_img = Image.open(img_path).convert("L")
        processed_img, meta = resize_with_padding_image(original_img, IMG_SIZE)
        x = prepare_input(processed_img)
        pred = model.predict(x, verbose=0)[0]

        prob_A = pred[:, :, 0]
        prob_B = pred[:, :, 1]
        prob_C = pred[:, :, 2]

        mask_A_256 = clean_mask(prob_A > args.thresh_a, keep_components=1)
        mask_B_256 = clean_mask(prob_B > args.thresh_b, keep_components=1)
        mask_C_256 = clean_mask(prob_C > args.thresh_c, keep_components=1)

        mask_A = restore_mask_to_original(mask_A_256, meta)
        mask_B = restore_mask_to_original(mask_B_256, meta)
        mask_C = restore_mask_to_original(mask_C_256, meta)

        # Save processed input 256 để debug đúng input model.
        processed_img.save(out["processed_images"] / name)

        # Save masks original size.
        save_binary_mask(mask_A, out["masks_A"] / name)
        save_binary_mask(mask_B, out["masks_B"] / name)
        save_binary_mask(mask_C, out["masks_C"] / name)

        # Save overlay/contour original size.
        overlay = make_overlay(original_img, mask_A, mask_B, mask_C)
        overlay.save(out["overlays"] / name)
        contour_preview = make_contour_preview(original_img, mask_A, mask_B, mask_C)
        contour_preview.save(out["contours"] / name)

        # Debug visualization vẫn dùng 256 cho nhẹ.
        overlay_256 = make_overlay(processed_img, mask_A_256, mask_B_256, mask_C_256)
        contour_256 = make_contour_preview(processed_img, mask_A_256, mask_B_256, mask_C_256)
        make_visualization(
            base_img=processed_img,
            overlay_img=overlay_256,
            contour_img=contour_256,
            mask_A=mask_A_256,
            mask_B=mask_B_256,
            mask_C=mask_C_256,
            save_path=out["visualizations"] / name,
        )

        # Save 2 file A/B cùng thư mục theo tên gốc + fill vùng bị đè C.
        save_separated_ab(
            original_gray=original_img,
            stem=stem,
            mask_A=mask_A,
            mask_B=mask_B,
            mask_C=mask_C,
            out_dir=out["separated"],
            transparent=not args.white_background,
        )

        if idx % 20 == 0 or idx == len(image_paths):
            print(f"Predicted {idx}/{len(image_paths)}")

    print("Done real inference + A/B separation.")
    print(f"Results saved to: {output_dir}")
    print(f"A/B separated files: {out['separated']}")


if __name__ == "__main__":
    main()
