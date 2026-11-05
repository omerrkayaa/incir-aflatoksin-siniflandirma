from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_split"
OUTPUT_DIR = BASE_DIR / "review_sheets"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

THUMB_SIZE = (160, 120)
TEXT_HEIGHT = 35
CELL_WIDTH = 180
CELL_HEIGHT = 170

COLS = 5
ROWS = 6
IMAGES_PER_SHEET = COLS * ROWS


def get_images(folder):
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def make_sheet(images, output_path, title):
    sheet_width = COLS * CELL_WIDTH
    sheet_height = ROWS * CELL_HEIGHT + 50

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    draw.text((10, 10), title, fill="black")

    for idx, img_path in enumerate(images):
        row = idx // COLS
        col = idx % COLS

        x = col * CELL_WIDTH
        y = row * CELL_HEIGHT + 50

        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail(THUMB_SIZE)

            img_x = x + (CELL_WIDTH - img.width) // 2
            img_y = y

            sheet.paste(img, (img_x, img_y))

            filename = img_path.name
            if len(filename) > 22:
                filename = filename[:19] + "..."

            draw.text((x + 5, y + THUMB_SIZE[1] + 5), filename, fill="black")

        except Exception as e:
            draw.text((x + 5, y + 40), "BOZUK DOSYA", fill="red")
            draw.text((x + 5, y + 60), img_path.name[:20], fill="red")

    sheet.save(output_path)


def process_folder(split, label):
    folder = DATASET_DIR / split / label

    if not folder.exists():
        print(f"Klasör bulunamadı: {folder}")
        return

    images = get_images(folder)
    images.sort()

    output_folder = OUTPUT_DIR / split / label
    output_folder.mkdir(parents=True, exist_ok=True)

    total_sheets = math.ceil(len(images) / IMAGES_PER_SHEET)

    print(f"{split}/{label}: {len(images)} görüntü, {total_sheets} sayfa oluşturulacak.")

    for i in range(total_sheets):
        start = i * IMAGES_PER_SHEET
        end = start + IMAGES_PER_SHEET

        batch = images[start:end]

        output_path = output_folder / f"{split}_{label}_sheet_{i+1}.jpg"
        title = f"{split}/{label} - Sheet {i+1}/{total_sheets}"

        make_sheet(batch, output_path, title)

        print(f"Oluşturuldu: {output_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for split in ["train", "val", "test"]:
        for label in ["healthy", "unhealthy"]:
            process_folder(split, label)

    print("\nİşlem tamamlandı.")
    print(f"Kontrol sayfaları burada: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()