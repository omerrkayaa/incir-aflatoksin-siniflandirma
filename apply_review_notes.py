import csv
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset_split"
NOTES_FILE = BASE_DIR / "review_notes.csv"
REMOVED_DIR = BASE_DIR / "removed_bad_images"
REPORT_FILE = BASE_DIR / "removed_images_report.csv"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

COLS = 5
ROWS = 6
IMAGES_PER_SHEET = COLS * ROWS


def get_images(folder):
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort()
    return images


def parse_sheet_number(sheet_name):
    stem = Path(sheet_name).stem
    parts = stem.split("_sheet_")

    if len(parts) != 2:
        raise ValueError(f"Sheet adi okunamadi: {sheet_name}")

    return int(parts[1])


def build_dataset_snapshot():
    """
    En kritik kisim burasi.
    Tum dosya listeleri en basta sabitleniyor.
    Boylece dosya tasindikca row/col indeksleri kaymiyor.
    """
    snapshot = {}

    for split in ["train", "val", "test"]:
        for label in ["healthy", "unhealthy"]:
            folder = DATASET_DIR / split / label

            if folder.exists():
                snapshot[(split, label)] = get_images(folder)
            else:
                snapshot[(split, label)] = []

    return snapshot


def get_image_from_snapshot(snapshot, split, label, sheet_name, row, col):
    images = snapshot[(split, label)]

    sheet_no = parse_sheet_number(sheet_name)

    row = int(row)
    col = int(col)

    local_index = (row - 1) * COLS + (col - 1)
    global_index = (sheet_no - 1) * IMAGES_PER_SHEET + local_index

    if global_index < 0 or global_index >= len(images):
        raise IndexError(
            f"Gorsel bulunamadi: {sheet_name}, row={row}, col={col}, index={global_index}"
        )

    return images[global_index]


def move_file_safely(src, split, label):
    target_folder = REMOVED_DIR / split / label
    target_folder.mkdir(parents=True, exist_ok=True)

    dst = target_folder / src.name

    if not dst.exists():
        shutil.move(str(src), str(dst))
        return dst

    stem = dst.stem
    suffix = dst.suffix
    counter = 1

    while True:
        new_dst = target_folder / f"{stem}_copy{counter}{suffix}"
        if not new_dst.exists():
            shutil.move(str(src), str(new_dst))
            return new_dst
        counter += 1


print("Review notlari sabit indeks yontemiyle uygulanıyor...")
print("--------------------------------")

if not NOTES_FILE.exists():
    print(f"HATA: review_notes.csv bulunamadi: {NOTES_FILE}")
    exit()

if not DATASET_DIR.exists():
    print(f"HATA: dataset_split klasoru bulunamadi: {DATASET_DIR}")
    exit()

snapshot = build_dataset_snapshot()

remove_targets = {}
skipped_records = []

with open(NOTES_FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for line_no, row_data in enumerate(reader, start=2):
        sheet = row_data["sheet"].strip()
        split = row_data["split"].strip()
        label = row_data["label"].strip()
        row = row_data["row"].strip()
        col = row_data["col"].strip()
        decision = row_data["decision"].strip().lower()
        note = row_data["note"].strip()

        if decision != "remove":
            continue

        if row.upper() == "ALL" or col.upper() == "ALL":
            skipped_records.append([
                line_no, sheet, split, label, row, col, decision,
                "ALL satiri remove icin atlandi"
            ])
            continue

        try:
            image_path = get_image_from_snapshot(
                snapshot, split, label, sheet, row, col
            )

            key = str(image_path)

            if key not in remove_targets:
                remove_targets[key] = {
                    "line_no": line_no,
                    "sheet": sheet,
                    "split": split,
                    "label": label,
                    "row": row,
                    "col": col,
                    "image_path": image_path,
                    "note": note
                }

        except Exception as e:
            skipped_records.append([
                line_no, sheet, split, label, row, col, decision, str(e)
            ])

removed_records = []

for item in remove_targets.values():
    image_path = item["image_path"]
    split = item["split"]
    label = item["label"]

    if not image_path.exists():
        skipped_records.append([
            item["line_no"],
            item["sheet"],
            split,
            label,
            item["row"],
            item["col"],
            "remove",
            "Dosya bulunamadi veya daha once tasinmis"
        ])
        continue

    new_path = move_file_safely(image_path, split, label)

    removed_records.append([
        item["line_no"],
        item["sheet"],
        split,
        label,
        item["row"],
        item["col"],
        image_path.name,
        str(new_path),
        item["note"]
    ])

    print(
        f"TASINDI: {split}/{label} | {item['sheet']} "
        f"row={item['row']} col={item['col']} -> {image_path.name}"
    )

print("--------------------------------")
print(f"Toplam tasinan goruntu: {len(removed_records)}")
print(f"Atlanan / hata alinan kayit: {len(skipped_records)}")

with open(REPORT_FILE, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "type",
        "line_no",
        "sheet",
        "split",
        "label",
        "row",
        "col",
        "image_name",
        "new_path_or_reason",
        "note"
    ])

    for rec in removed_records:
        line_no, sheet, split, label, row, col, image_name, new_path, note = rec
        writer.writerow([
            "removed",
            line_no,
            sheet,
            split,
            label,
            row,
            col,
            image_name,
            new_path,
            note
        ])

    for rec in skipped_records:
        line_no, sheet, split, label, row, col, decision, reason = rec
        writer.writerow([
            "skipped",
            line_no,
            sheet,
            split,
            label,
            row,
            col,
            "",
            reason,
            decision
        ])

print(f"Rapor dosyasi olusturuldu: {REPORT_FILE}")
print(f"Tasinan kotu gorseller burada: {REMOVED_DIR}")
print("Islem tamamlandi.")