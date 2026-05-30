from pathlib import Path
from PIL import Image
import numpy as np
import shutil

# =========================
# CONFIG
# =========================
ROOT = Path(__file__).resolve().parent
INPUT_IMAGE_DIR = ROOT / "generated_data" / "images"
INPUT_MASK_A_DIR = ROOT / "generated_data" / "masks_A"
INPUT_MASK_B_DIR = ROOT / "generated_data" / "masks_B"
INPUT_MASK_C_DIR = ROOT / "generated_data" / "masks_C"

OUTPUT_ROOT = ROOT / "processed_data_256"
OUTPUT_IMAGE_DIR = OUTPUT_ROOT / "images"
OUTPUT_MASK_A_DIR = OUTPUT_ROOT / "masks_A"
OUTPUT_MASK_B_DIR = OUTPUT_ROOT / "masks_B"
OUTPUT_MASK_C_DIR = OUTPUT_ROOT / "masks_C"

TARGET_SIZE = 256
CLEAR_OLD_OUTPUT = True

if CLEAR_OLD_OUTPUT and OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

for folder in [
    OUTPUT_IMAGE_DIR,
    OUTPUT_MASK_A_DIR,
    OUTPUT_MASK_B_DIR,
    OUTPUT_MASK_C_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# =========================
# HELPER
# =========================
def resize_with_padding(img, target_size=256, is_mask=False):
    """
    Resize giữ tỉ lệ, sau đó padding để thành target_size x target_size.
    image:
        - resize: bilinear
        - padding: 255
    mask:
        - resize: nearest
        - padding: 0
    """
    w, h = img.size

    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if is_mask:
        img = img.resize((new_w, new_h), Image.NEAREST)
        canvas = Image.new("L", (target_size, target_size), 0)
    else:
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("L", (target_size, target_size), 255)

    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    canvas.paste(img, (paste_x, paste_y))
    return canvas


def binarize_mask(mask_img):
    """
    Đảm bảo mask chỉ còn 0 và 1 rồi lưu ra 0 hoặc 255.
    """
    arr = np.array(mask_img)
    arr = (arr > 127).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


# =========================
# PROCESS
# =========================
image_paths = sorted(INPUT_IMAGE_DIR.glob("*.png"))

print(f"Found {len(image_paths)} generated images.")

if len(image_paths) == 0:
    raise FileNotFoundError("No generated images found. Run 3v1_generate_synthetic_masks.py first.")

for idx, img_path in enumerate(image_paths, start=1):
    name = img_path.name

    mask_A_path = INPUT_MASK_A_DIR / name
    mask_B_path = INPUT_MASK_B_DIR / name
    mask_C_path = INPUT_MASK_C_DIR / name

    if not (mask_A_path.exists() and mask_B_path.exists() and mask_C_path.exists()):
        print(f"Missing mask for {name}, skipping.")
        continue

    # -------- IMAGE --------
    img = Image.open(img_path).convert("L")   # grayscale
    img = resize_with_padding(img, target_size=TARGET_SIZE, is_mask=False)
    img.save(OUTPUT_IMAGE_DIR / name)

    # -------- MASK A --------
    mask_A = Image.open(mask_A_path).convert("L")
    mask_A = binarize_mask(mask_A)
    mask_A = resize_with_padding(mask_A, target_size=TARGET_SIZE, is_mask=True)
    mask_A = binarize_mask(mask_A)
    mask_A.save(OUTPUT_MASK_A_DIR / name)

    # -------- MASK B --------
    mask_B = Image.open(mask_B_path).convert("L")
    mask_B = binarize_mask(mask_B)
    mask_B = resize_with_padding(mask_B, target_size=TARGET_SIZE, is_mask=True)
    mask_B = binarize_mask(mask_B)
    mask_B.save(OUTPUT_MASK_B_DIR / name)

    # -------- MASK C --------
    mask_C = Image.open(mask_C_path).convert("L")
    mask_C = binarize_mask(mask_C)
    mask_C = resize_with_padding(mask_C, target_size=TARGET_SIZE, is_mask=True)
    mask_C = binarize_mask(mask_C)
    mask_C.save(OUTPUT_MASK_C_DIR / name)

    if idx % 100 == 0:
        print(f"Processed {idx}/{len(image_paths)}")

print("Done preprocessing to 256x256.")