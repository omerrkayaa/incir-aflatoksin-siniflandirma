from pathlib import Path
from PIL import Image, ImageOps
import shutil
import csv

# =========================
# AYARLAR
# =========================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "dataset_split"
OUTPUT_DIR = BASE_DIR / "dataset_processed"

REPORT_FILE = BASE_DIR / "preprocess_report.csv"

TARGET_SIZE = (224, 224)

# 0.85
#  = merkeze doğru hafif kırpma
# Eğer incirler kesiliyor gibi olursa bunu 0.95 yapabiliriz.
CROP_RATIO = 0.85

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# =========================
# FONKSİYONLAR
# =========================

def center_crop_square(image, crop_ratio=0.90):
    """
    Görseli önce merkezden kare yapar.
    Sonra kenarlardan biraz daha daraltır.
    Amaç: kenardaki el/yabancı nesne/siyah köşe etkisini azaltmak.
    """
    width, height = image.size

    side = min(width, height)

    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side

    image = image.crop((left, top, right, bottom))

    if crop_ratio < 1.0:
        width, height = image.size
        new_side = int(min(width, height) * crop_ratio)

        left = (width - new_side) // 2
        top = (height - new_side) // 2
        right = left + new_side
        bottom = top + new_side

        image = image.crop((left, top, right, bottom))

    return image


def process_image(input_path, output_path):
    image = Image.open(input_path)

    # Telefonla çekilmiş fotoğraflarda EXIF yön bilgisini düzeltir
    image = ImageOps.exif_transpose(image)

    # Model için RGB formatına çevir
    image = image.convert("RGB")

    # Crop
    image = center_crop_square(image, CROP_RATIO)

    # Resize
    image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Hepsini jpg olarak kaydediyoruz
    image.save(output_path, quality=95, optimize=True)


def get_image_files(folder):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    files = sorted(set(files))
    return files


# =========================
# ANA PROGRAM
# =========================

print("Preprocessing basladi...")
print("--------------------------------")

if not INPUT_DIR.exists():
    print(f"HATA: dataset_split klasoru bulunamadi: {INPUT_DIR}")
    exit()

if OUTPUT_DIR.exists():
    print(f"Eski dataset_processed klasoru siliniyor: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)

records = []
total_processed = 0
total_failed = 0

for split in ["train", "val", "test"]:
    for label in ["healthy", "unhealthy"]:
        input_folder = INPUT_DIR / split / label
        output_folder = OUTPUT_DIR / split / label

        if not input_folder.exists():
            print(f"Klasor bulunamadi, atlandi: {input_folder}")
            continue

        images = get_image_files(input_folder)

        print(f"{split}/{label} isleniyor. Toplam: {len(images)}")

        for i, image_path in enumerate(images, start=1):
            try:
                output_path = output_folder / f"{image_path.stem}.jpg"
                process_image(image_path, output_path)

                records.append([
                    "processed",
                    split,
                    label,
                    image_path.name,
                    output_path.name,
                    str(input_folder),
                    str(output_folder),
                    ""
                ])

                total_processed += 1

                if i % 50 == 0 or i == len(images):
                    print(f"{split}/{label}: {i}/{len(images)} islem tamamlandi.")

            except Exception as e:
                records.append([
                    "failed",
                    split,
                    label,
                    image_path.name,
                    "",
                    str(input_folder),
                    str(output_folder),
                    str(e)
                ])

                total_failed += 1
                print(f"HATA: {image_path.name} -> {e}")

print("--------------------------------")
print(f"Toplam islenen gorsel: {total_processed}")
print(f"Hata alinan gorsel: {total_failed}")
print(f"Cikti klasoru: {OUTPUT_DIR}")

with open(REPORT_FILE, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "status",
        "split",
        "label",
        "input_file",
        "output_file",
        "input_folder",
        "output_folder",
        "error"
    ])
    writer.writerows(records)

print(f"Rapor dosyasi olusturuldu: {REPORT_FILE}")
print("Preprocessing tamamlandi.")