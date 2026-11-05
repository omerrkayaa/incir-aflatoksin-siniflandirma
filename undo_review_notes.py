import csv
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_split"
REPORT_FILE = BASE_DIR / "removed_images_report.csv"

print("Geri alma islemi basladi...")

if not REPORT_FILE.exists():
    print(f"HATA: Rapor dosyasi bulunamadi: {REPORT_FILE}")
    exit()

restored = 0
skipped = 0

with open(REPORT_FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["type"] != "removed":
            continue

        split = row["split"]
        label = row["label"]
        image_name = row["image_name"]
        old_removed_path = Path(row["new_path_or_reason"])

        target_folder = DATASET_DIR / split / label
        target_folder.mkdir(parents=True, exist_ok=True)

        target_path = target_folder / image_name

        if not old_removed_path.exists():
            print(f"ATLANDI, dosya yok: {old_removed_path}")
            skipped += 1
            continue

        if target_path.exists():
            print(f"ATLANDI, hedefte zaten var: {target_path}")
            skipped += 1
            continue

        shutil.move(str(old_removed_path), str(target_path))
        restored += 1
        print(f"GERI ALINDI: {image_name}")

print("--------------------------------")
print(f"Geri alinan dosya: {restored}")
print(f"Atlanan dosya: {skipped}")
print("Geri alma tamamlandi.")