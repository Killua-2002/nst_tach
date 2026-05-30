from pathlib import Path
import shutil
import random

CLEAR_OLD_DATASET = True

# =========================
# CONFIG
# =========================
ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "processed_data_256" / "images"
MASK_A_DIR = ROOT / "processed_data_256" / "masks_A"
MASK_B_DIR = ROOT / "processed_data_256" / "masks_B"
MASK_C_DIR = ROOT / "processed_data_256" / "masks_C"

DATASET_DIR = ROOT / "dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)

# =========================
# INIT
# =========================
if CLEAR_OLD_DATASET and DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)

# =========================
# CREATE FOLDERS
# =========================
folders = [
    "train/images", "train/masks_A", "train/masks_B", "train/masks_C",
    "val/images", "val/masks_A", "val/masks_B", "val/masks_C",
    "test/images", "test/masks_A", "test/masks_B", "test/masks_C",
]

for folder in folders:
    (DATASET_DIR / folder).mkdir(parents=True, exist_ok=True)

# =========================
# LOAD FILES
# =========================
image_paths = sorted(IMAGE_DIR.glob("*.png"))
names = [p.name for p in image_paths]

if len(names) == 0:
    raise FileNotFoundError("No processed images found. Run 4v1_preprocess_to_256.py first.")

random.shuffle(names)

n = len(names)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)
n_test = n - n_train - n_val

train_names = names[:n_train]
val_names = names[n_train:n_train + n_val]
test_names = names[n_train + n_val:]

print(f"Total: {n}")
print(f"Train: {len(train_names)}")
print(f"Val: {len(val_names)}")
print(f"Test: {len(test_names)}")

# =========================
# COPY FUNCTION
# =========================
def copy_set(file_names, split_name):
    for name in file_names:
        shutil.copy2(IMAGE_DIR / name, DATASET_DIR / split_name / "images" / name)
        shutil.copy2(MASK_A_DIR / name, DATASET_DIR / split_name / "masks_A" / name)
        shutil.copy2(MASK_B_DIR / name, DATASET_DIR / split_name / "masks_B" / name)
        shutil.copy2(MASK_C_DIR / name, DATASET_DIR / split_name / "masks_C" / name)

copy_set(train_names, "train")
copy_set(val_names, "val")
copy_set(test_names, "test")

print("Done splitting dataset.")