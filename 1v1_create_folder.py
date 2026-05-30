from pathlib import Path

ROOT = Path(__file__).resolve().parent

folders = [
    "source_data/overlap_raw",
    "source_data/single_chromosomes",

    "generated_data/images",
    "generated_data/masks_A",
    "generated_data/masks_B",
    "generated_data/masks_C",
    "generated_data/previews",

    "dataset/train/images",
    "dataset/train/masks_A",
    "dataset/train/masks_B",
    "dataset/train/masks_C",

    "dataset/val/images",
    "dataset/val/masks_A",
    "dataset/val/masks_B",
    "dataset/val/masks_C",

    "dataset/test/images",
    "dataset/test/masks_A",
    "dataset/test/masks_B",
    "dataset/test/masks_C",

    "results/logs",
    "results/predicted_masks",
    "results/predicted_contours",
    "results/metric_reports",
]

for folder in folders:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

print("Done creating project folders.")