from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent

DATASET_SPLIT = BASE_DIR / "dataset_split"
DATASET_PROCESSED = BASE_DIR / "dataset_processed"
REMOVED_DIR = BASE_DIR / "removed_bad_images"
REVIEW_NOTES = BASE_DIR / "review_notes.csv"

OUTPUT_CSV = BASE_DIR / "dataset_statistics.csv"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def count_images(folder):
    if not folder.exists():
        return 0

    total = 0
    for ext in IMAGE_EXTENSIONS:
        total += len(list(folder.glob(f"*{ext}")))
        total += len(list(folder.glob(f"*{ext.upper()}")))
    return total


def count_review_notes():
    if not REVIEW_NOTES.exists():
        return {
            "review_total": 0,
            "remove": 0,
            "crop_candidate": 0,
            "keep": 0,
        }

    counts = {
        "review_total": 0,
        "remove": 0,
        "crop_candidate": 0,
        "keep": 0,
    }

    with open(REVIEW_NOTES, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts["review_total"] += 1
            decision = row["decision"].strip().lower()

            if decision in counts:
                counts[decision] += 1

    return counts


rows = []

for dataset_name, dataset_path in [
    ("dataset_split", DATASET_SPLIT),
    ("dataset_processed", DATASET_PROCESSED),
    ("removed_bad_images", REMOVED_DIR),
]:
    for split in ["train", "val", "test"]:
        for label in ["healthy", "unhealthy"]:
            folder = dataset_path / split / label
            count = count_images(folder)

            rows.append([
                dataset_name,
                split,
                label,
                count
            ])


review_counts = count_review_notes()

with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(["dataset", "split", "label", "image_count"])
    writer.writerows(rows)

    writer.writerow([])
    writer.writerow(["review_notes_summary"])
    writer.writerow(["review_total", review_counts["review_total"]])
    writer.writerow(["remove", review_counts["remove"]])
    writer.writerow(["crop_candidate", review_counts["crop_candidate"]])
    writer.writerow(["keep", review_counts["keep"]])

print("Dataset istatistikleri:")
print("--------------------------------")

for row in rows:
    dataset_name, split, label, count = row
    print(f"{dataset_name:18} | {split:5} | {label:9} | {count}")

print("--------------------------------")
print("Review notes:")
print(f"Toplam not: {review_counts['review_total']}")
print(f"Remove: {review_counts['remove']}")
print(f"Crop candidate: {review_counts['crop_candidate']}")
print(f"Keep: {review_counts['keep']}")
print("--------------------------------")
print(f"Rapor olusturuldu: {OUTPUT_CSV}")   