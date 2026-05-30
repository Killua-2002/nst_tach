from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import shutil

# =========================
# CONFIG
# =========================

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "source_data" / "single_chromosomes"

OUTPUT_ROOT = ROOT / "prepared_single_chromosomes"
OUT_RGBA_DIR = OUTPUT_ROOT / "images_rgba"
OUT_MASK_DIR = OUTPUT_ROOT / "masks"
OUT_PREVIEW_DIR = OUTPUT_ROOT / "previews"

# Nếu muốn mỗi lần chạy xóa output cũ
CLEAR_OLD_OUTPUT = True

# Ngưỡng nền trắng
# bg_ref thường gần 255, threshold = bg_ref - BG_MARGIN
# tăng BG_MARGIN => ăn sâu hơn vào viền trắng
BG_MARGIN = 28

# Dọn nhiễu
MIN_OBJECT_AREA = 120

# Crop sát object
CROP_PADDING = 2

# Co mask vào 1 chút để giảm halo trắng ngoài rìa
# 0 = không co
# 1 = co nhẹ (khuyên dùng)
MASK_SHRINK_ITER = 1

# Morphology
KERNEL_SIZE = 3


# =========================
# INIT
# =========================

if CLEAR_OLD_OUTPUT and OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

for folder in [OUT_RGBA_DIR, OUT_MASK_DIR, OUT_PREVIEW_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================
# HELPERS
# =========================

def keep_largest_component(mask_bool):
    """
    Giữ component lớn nhất.
    """
    mask_uint8 = mask_bool.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    if num_labels <= 1:
        return mask_bool

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    return labels == largest_label


def get_border_background(gray):
    """
    Tìm nền trắng bằng cách:
    - lấy giá trị biên ảnh
    - chỉ coi nền là vùng sáng và có kết nối với border
    => giữ lại được các vùng trắng nằm BÊN TRONG NST
    """
    h, w = gray.shape

    border_pixels = np.concatenate([
        gray[0, :],
        gray[h - 1, :],
        gray[:, 0],
        gray[:, w - 1]
    ])

    bg_ref = np.median(border_pixels)
    threshold = max(210, bg_ref - BG_MARGIN)

    # pixel rất sáng mới có khả năng là nền
    candidate_bg = gray >= threshold

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_bg.astype(np.uint8),
        connectivity=8
    )

    # label xuất hiện ở border => background thật
    border_labels = set(np.unique(np.concatenate([
        labels[0, :],
        labels[h - 1, :],
        labels[:, 0],
        labels[:, w - 1]
    ])))

    # bỏ label 0 (phần không phải candidate_bg)
    border_labels.discard(0)

    if len(border_labels) == 0:
        return np.zeros_like(gray, dtype=bool), threshold, bg_ref

    background = np.isin(labels, list(border_labels)) & candidate_bg
    return background, threshold, bg_ref


def extract_object_rgba(img_path):
    """
    Tách NST khỏi nền trắng.
    Giữ:
    - texture bên trong
    - các vùng trắng bên trong NST
    Xóa:
    - nền trắng nối với biên ảnh
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    gray = np.array(img.convert("L"))

    background, threshold, bg_ref = get_border_background(gray)

    # object = phần KHÔNG phải background
    object_mask = ~background

    # clean
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    object_mask_uint8 = object_mask.astype(np.uint8) * 255

    object_mask_uint8 = cv2.morphologyEx(object_mask_uint8, cv2.MORPH_OPEN, kernel)
    object_mask_uint8 = cv2.morphologyEx(object_mask_uint8, cv2.MORPH_CLOSE, kernel)

    object_mask = object_mask_uint8 > 0
    object_mask = keep_largest_component(object_mask)

    if object_mask.sum() < MIN_OBJECT_AREA:
        return None, None, None

    # Co mask nhẹ để ăn vào sát hơn, giảm viền trắng giả
    if MASK_SHRINK_ITER > 0:
        eroded = cv2.erode(
            (object_mask.astype(np.uint8) * 255),
            kernel,
            iterations=MASK_SHRINK_ITER
        )
        object_mask = eroded > 0

    if object_mask.sum() < MIN_OBJECT_AREA:
        return None, None, None

    # Crop sát object
    ys, xs = np.where(object_mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - CROP_PADDING)
    y1 = max(0, y1 - CROP_PADDING)
    x2 = min(gray.shape[1] - 1, x2 + CROP_PADDING)
    y2 = min(gray.shape[0] - 1, y2 + CROP_PADDING)

    cropped_rgb = arr[y1:y2+1, x1:x2+1]
    cropped_mask = object_mask[y1:y2+1, x1:x2+1]

    rgba = np.zeros((cropped_rgb.shape[0], cropped_rgb.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = cropped_rgb
    rgba[:, :, 3] = cropped_mask.astype(np.uint8) * 255

    mask_img = (cropped_mask.astype(np.uint8) * 255)

    rgba_img = Image.fromarray(rgba, mode="RGBA")
    mask_img = Image.fromarray(mask_img, mode="L")

    info = {
        "bg_ref": float(bg_ref),
        "threshold": float(threshold),
        "area": int(object_mask.sum())
    }

    return rgba_img, mask_img, info


def make_preview(rgba_img, mask_img):
    """
    Preview để kiểm tra bằng mắt:
    - nền xám nhạt
    - object sạch
    - contour đỏ
    """
    bg = Image.new("RGB", rgba_img.size, (240, 240, 240))
    bg.paste(rgba_img, mask=rgba_img.split()[-1])

    preview = bg.copy()
    draw = ImageDraw.Draw(preview)

    mask_arr = np.array(mask_img)
    contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim == 2 and len(cnt) >= 2:
            pts = [tuple(map(int, p)) for p in cnt]
            draw.line(pts + [pts[0]], fill=(255, 0, 0), width=1)

    return preview


# =========================
# MAIN
# =========================

def main():
    image_paths = sorted(INPUT_DIR.glob("*.png"))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No PNG files found in {INPUT_DIR}")

    print(f"Found {len(image_paths)} single chromosome images.")
    print("Processing...")

    saved = 0

    for idx, img_path in enumerate(image_paths, start=1):
        rgba_img, mask_img, info = extract_object_rgba(img_path)

        if rgba_img is None:
            print(f"[SKIP] {img_path.name} -> object too small / extraction failed")
            continue

        out_name = img_path.stem + ".png"

        rgba_img.save(OUT_RGBA_DIR / out_name)
        mask_img.save(OUT_MASK_DIR / out_name)

        preview = make_preview(rgba_img, mask_img)
        preview.save(OUT_PREVIEW_DIR / out_name)

        saved += 1

        if idx % 100 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)} | Saved: {saved}")

    print("Done.")
    print(f"Saved cleaned RGBA objects to: {OUT_RGBA_DIR}")
    print(f"Saved masks to: {OUT_MASK_DIR}")
    print(f"Saved previews to: {OUT_PREVIEW_DIR}")


if __name__ == "__main__":
    main()