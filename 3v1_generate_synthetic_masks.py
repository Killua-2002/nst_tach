from pathlib import Path
from PIL import Image
import numpy as np
import random
import cv2
import shutil

# =========================
# CONFIG
# =========================

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "prepared_single_chromosomes" / "images_rgba"

OUT_IMAGE_DIR = ROOT / "generated_data" / "images"
OUT_MASK_A_DIR = ROOT / "generated_data" / "masks_A"
OUT_MASK_B_DIR = ROOT / "generated_data" / "masks_B"
OUT_MASK_C_DIR = ROOT / "generated_data" / "masks_C"
OUT_PREVIEW_DIR = ROOT / "generated_data" / "previews"

CANVAS_SIZE = 512
NUM_SAMPLES = 3000

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Object extraction
BACKGROUND_DIFF_THRESHOLD = 22
MIN_OBJECT_AREA = 100

# Overlap control
MIN_OVERLAP_PIXELS = 120
MAX_OVERLAP_RATIO = 0.55

# Size control
TARGET_LONG_SIDE_MIN = 250
TARGET_LONG_SIDE_MAX = 380

# Nếu muốn xóa output cũ mỗi lần chạy
CLEAR_OLD_OUTPUT = True


# =========================
# INIT FOLDERS
# =========================

if CLEAR_OLD_OUTPUT and (ROOT / "generated_data").exists():
    shutil.rmtree(ROOT / "generated_data")

for folder in [
    OUT_IMAGE_DIR,
    OUT_MASK_A_DIR,
    OUT_MASK_B_DIR,
    OUT_MASK_C_DIR,
    OUT_PREVIEW_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================
# HELPER FUNCTIONS
# =========================

def keep_largest_component(mask):
    """
    Giữ object lớn nhất, bỏ noise nhỏ.
    mask: bool array
    """
    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    if num_labels <= 1:
        return mask

    # label 0 là background
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    largest = labels == largest_label
    return largest


def extract_chromosome_object(image_path):
    """
    Tách NST khỏi nền.
    Output:
    - obj_rgba: ảnh NST nền trong suốt
    - obj_mask: mask NST 0/255
    """
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    h, w = rgb.shape[:2]

    # Nếu ảnh có alpha sẵn thì dùng alpha
    if np.min(alpha) < 250:
        mask = alpha > 10
    else:
        # Ước lượng màu nền từ 4 góc ảnh
        corner_size = max(5, min(h, w) // 12)

        corners = np.concatenate([
            rgb[:corner_size, :corner_size].reshape(-1, 3),
            rgb[:corner_size, w-corner_size:w].reshape(-1, 3),
            rgb[h-corner_size:h, :corner_size].reshape(-1, 3),
            rgb[h-corner_size:h, w-corner_size:w].reshape(-1, 3),
        ], axis=0)

        bg_color = np.median(corners, axis=0)

        # Pixel khác nền đủ nhiều thì là NST
        diff = np.linalg.norm(rgb.astype(np.float32) - bg_color.astype(np.float32), axis=2)
        mask = diff > BACKGROUND_DIFF_THRESHOLD

        # Nếu ảnh grayscale nền trắng, NST tối hơn nền
        gray = np.mean(rgb, axis=2)
        bg_gray = np.mean(bg_color)
        dark_object = gray < bg_gray - 8

        mask = mask | dark_object

    # Clean mask
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)

    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    mask = mask_uint8 > 0
    mask = keep_largest_component(mask)

    if mask.sum() < MIN_OBJECT_AREA:
        return None, None

    # Crop sát object
    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad = 6
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    cropped_rgb = rgb[y1:y2+1, x1:x2+1]
    cropped_mask = mask[y1:y2+1, x1:x2+1]

    # Tạo RGBA object: nền trong suốt, chỉ giữ thân NST
    obj_rgba_arr = np.zeros((cropped_rgb.shape[0], cropped_rgb.shape[1], 4), dtype=np.uint8)
    obj_rgba_arr[:, :, :3] = cropped_rgb
    obj_rgba_arr[:, :, 3] = cropped_mask.astype(np.uint8) * 255

    obj_mask_arr = cropped_mask.astype(np.uint8) * 255

    obj_rgba = Image.fromarray(obj_rgba_arr, mode="RGBA")
    obj_mask = Image.fromarray(obj_mask_arr, mode="L")

    return obj_rgba, obj_mask


def resize_keep_ratio(img, mask, target_long_side):
    """
    Resize object và mask giữ tỉ lệ.
    """
    w, h = img.size
    scale = target_long_side / max(w, h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    mask_resized = mask.resize((new_w, new_h), Image.NEAREST)

    return img_resized, mask_resized


def rotate_pair(img, mask, angle):
    """
    Xoay object và mask cùng góc.
    Object fill transparent, mask fill 0.
    """
    img_rot = img.rotate(
        angle,
        expand=True,
        resample=Image.BILINEAR,
        fillcolor=(255, 255, 255, 0)
    )

    mask_rot = mask.rotate(
        angle,
        expand=True,
        resample=Image.NEAREST,
        fillcolor=0
    )

    return img_rot, mask_rot


def paste_object(canvas_rgba, canvas_mask, obj_rgba, obj_mask, center_x, center_y):
    """
    Paste object RGBA vào canvas.
    Chỉ object được paste, nền trong suốt không dính vào ảnh.
    """
    obj_rgba = obj_rgba.convert("RGBA")
    obj_mask = obj_mask.convert("L")

    w, h = obj_rgba.size

    x = int(center_x - w / 2)
    y = int(center_y - h / 2)

    # Paste ảnh object bằng alpha thật
    canvas_rgba.alpha_composite(obj_rgba, dest=(x, y))

    # Paste mask tương ứng
    canvas_mask_arr = np.array(canvas_mask)
    obj_mask_arr = np.array(obj_mask)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(CANVAS_SIZE, x + w)
    y2 = min(CANVAS_SIZE, y + h)

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    if x1 < x2 and y1 < y2:
        region = canvas_mask_arr[y1:y2, x1:x2]
        obj_region = obj_mask_arr[oy1:oy2, ox1:ox2]
        region[obj_region > 0] = 255
        canvas_mask_arr[y1:y2, x1:x2] = region

    return canvas_rgba, Image.fromarray(canvas_mask_arr, mode="L")


def make_preview(image, mask_A, mask_B, mask_C):
    """
    Preview màu:
    A = đỏ
    B = xanh lá
    C = vàng
    """
    base = np.array(image.convert("RGB")).astype(np.float32)

    A = np.array(mask_A) > 0
    B = np.array(mask_B) > 0
    C = np.array(mask_C) > 0

    overlay = base.copy()

    overlay[A] = overlay[A] * 0.45 + np.array([255, 0, 0]) * 0.55
    overlay[B] = overlay[B] * 0.45 + np.array([0, 255, 0]) * 0.55
    overlay[C] = overlay[C] * 0.25 + np.array([255, 255, 0]) * 0.75

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def make_realistic_overlap_canvas(obj_A, mask_A, obj_B, mask_B):
    """
    Tạo canvas:
    - A xoay gần ngang
    - B gần dọc
    - tâm hai NST đặt gần nhau để phần giữa overlap
    """
    # Resize object
    target_A = random.randint(TARGET_LONG_SIDE_MIN, TARGET_LONG_SIDE_MAX)
    target_B = random.randint(TARGET_LONG_SIDE_MIN, TARGET_LONG_SIDE_MAX)

    obj_A, mask_A = resize_keep_ratio(obj_A, mask_A, target_A)
    obj_B, mask_B = resize_keep_ratio(obj_B, mask_B, target_B)

    # Vì single chromosome đang đứng:
    # A xoay gần ngang, B giữ gần dọc
    angle_A = 90 + random.uniform(-8, 8)
    angle_B = random.uniform(-8, 8)

    obj_A, mask_A = rotate_pair(obj_A, mask_A, angle_A)
    obj_B, mask_B = rotate_pair(obj_B, mask_B, angle_B)

    # Canvas trắng
    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))
    mask_A_canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    mask_B_canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)

    center_x = CANVAS_SIZE // 2 + random.randint(-15, 15)
    center_y = CANVAS_SIZE // 2 + random.randint(-15, 15)

    # Ép hai tâm gần nhau để phần giữa lồng nhau
    # A lệch trái/phải nhẹ, B lệch trên/dưới nhẹ
    A_cx = center_x + random.randint(-18, 18)
    A_cy = center_y + random.randint(-10, 10)

    B_cx = center_x + random.randint(-10, 10)
    B_cy = center_y + random.randint(-18, 18)

    # Random thứ tự paste để có biến thiên thị giác
    # Nhưng mask A/B vẫn giữ riêng.
    if random.random() < 0.5:
        canvas, mask_A_canvas = paste_object(canvas, mask_A_canvas, obj_A, mask_A, A_cx, A_cy)
        canvas, mask_B_canvas = paste_object(canvas, mask_B_canvas, obj_B, mask_B, B_cx, B_cy)
    else:
        canvas, mask_B_canvas = paste_object(canvas, mask_B_canvas, obj_B, mask_B, B_cx, B_cy)
        canvas, mask_A_canvas = paste_object(canvas, mask_A_canvas, obj_A, mask_A, A_cx, A_cy)

    A_arr = np.array(mask_A_canvas) > 0
    B_arr = np.array(mask_B_canvas) > 0
    C_arr = A_arr & B_arr

    overlap_pixels = int(C_arr.sum())
    area_A = max(1, int(A_arr.sum()))
    area_B = max(1, int(B_arr.sum()))

    overlap_ratio = overlap_pixels / min(area_A, area_B)

    if overlap_pixels < MIN_OVERLAP_PIXELS:
        return None

    if overlap_ratio > MAX_OVERLAP_RATIO:
        return None

    mask_C_canvas = Image.fromarray((C_arr.astype(np.uint8) * 255), mode="L")

    final_image = canvas.convert("RGB")

    # Có thể thêm noise rất nhẹ để giống ảnh thật hơn
    # Nhưng không làm quá mạnh.
    if random.random() < 0.35:
        arr = np.array(final_image).astype(np.int16)
        noise = np.random.normal(0, 2, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        final_image = Image.fromarray(arr, mode="RGB")

    return final_image, mask_A_canvas, mask_B_canvas, mask_C_canvas


# =========================
# MAIN GENERATION
# =========================

def main():
    single_paths = sorted(SOURCE_DIR.glob("*.png"))

    if len(single_paths) < 2:
        raise ValueError(f"Need at least 2 PNG images in {SOURCE_DIR}")

    print(f"Found {len(single_paths)} single chromosome PNG images.")
    print(f"Generating {NUM_SAMPLES} synthetic overlap samples...")

    created = 0
    attempts = 0
    max_attempts = NUM_SAMPLES * 50

    while created < NUM_SAMPLES and attempts < max_attempts:
        attempts += 1

        path_A, path_B = random.sample(single_paths, 2)

        obj_A, mask_A = extract_chromosome_object(path_A)
        obj_B, mask_B = extract_chromosome_object(path_B)

        if obj_A is None or obj_B is None:
            continue

        result = make_realistic_overlap_canvas(obj_A, mask_A, obj_B, mask_B)

        if result is None:
            continue

        final_image, mask_A_canvas, mask_B_canvas, mask_C_canvas = result

        created += 1
        name = f"img_{created:06d}.png"

        final_image.save(OUT_IMAGE_DIR / name)
        mask_A_canvas.save(OUT_MASK_A_DIR / name)
        mask_B_canvas.save(OUT_MASK_B_DIR / name)
        mask_C_canvas.save(OUT_MASK_C_DIR / name)

        preview = make_preview(final_image, mask_A_canvas, mask_B_canvas, mask_C_canvas)
        preview.save(OUT_PREVIEW_DIR / name)

        if created % 100 == 0:
            print(f"Created {created}/{NUM_SAMPLES}")

    print("Done.")
    print(f"Created: {created}")
    print(f"Attempts: {attempts}")

    if created < NUM_SAMPLES:
        print("Warning: Could not create enough samples.")
        print("Try lowering MIN_OVERLAP_PIXELS or increasing max_attempts.")


if __name__ == "__main__":
    main()