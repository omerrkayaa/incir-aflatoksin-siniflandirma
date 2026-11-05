from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
import shutil

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset_processed"
OUTPUT_DIR = BASE_DIR / "review_processed_sheets"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

COLS = 5
ROWS = 6
IMAGES_PER_SHEET = COLS * ROWS

THUMB_W = 160
THUMB_H = 160

MARGIN = 25
GAP_X = 20
GAP_Y = 35
TITLE_H = 35
TEXT_H = 25


def get_images(folder):
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    images = sorted(set(images))
    return images


def make_sheet(images, output_path, title):
    sheet_w = MARGIN * 2 + COLS * THUMB_W + (COLS - 1) * GAP_X
    sheet_h = MARGIN * 2 + TITLE_H + ROWS * (THUMB_H + TEXT_H) + (ROWS - 1) * GAP_Y

    canvas = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_text = ImageFont.truetype("arial.ttf", 10)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    draw.text((MARGIN, 10), title, fill="black", font=font_title)

    for idx, img_path in enumerate(images):
        row = idx // COLS
        col = idx % COLS

        x = MARGIN + col * (THUMB_W + GAP_X)
        y = MARGIN + TITLE_H + row * (THUMB_H + TEXT_H + GAP_Y)

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)

            canvas.paste(img, (x, y))

            name = img_path.name
            if len(name) > 24:
                name = name[:21] + "..."

            draw.text((x, y + THUMB_H + 3), name, fill="black", font=font_text)

        except Exception as e:
            draw.rectangle((x, y, x + THUMB_W, y + THUMB_H), outline="red", width=3)
            draw.text((x, y), "HATA", fill="red", font=font_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def process_folder(split, label):
    input_folder = DATASET_DIR / split / label
    output_folder = OUTPUT_DIR / split / label

    if not input_folder.exists():
        print(f"Klasor yok, atlandi: {input_folder}")
        return

    images = get_images(input_folder)

    if not images:
        print(f"Gorsel yok: {input_folder}")
        return

    total_sheets = math.ceil(len(images) / IMAGES_PER_SHEET)

    print(f"{split}/{label}: {len(images)} gorsel, {total_sheets} sayfa olusturulacak.")

    for i in range(total_sheets):
        start = i * IMAGES_PER_SHEET
        end = start + IMAGES_PER_SHEET
        batch = images[start:end]

        output_path = output_folder / f"{split}_{label}_processed_sheet_{i+1}.jpg"
        title = f"{split}/{label} processed - Sheet {i+1}/{total_sheets}"

        make_sheet(batch, output_path, title)

        print(f"Olusturuldu: {output_path}")


def main():
    print("Processed review sheet olusturma basladi...")
    print("--------------------------------")

    if not DATASET_DIR.exists():
        print(f"HATA: dataset_processed bulunamadi: {DATASET_DIR}")
        return

    if OUTPUT_DIR.exists():
        print(f"Eski review_processed_sheets siliniyor: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        for label in ["healthy", "unhealthy"]:
            process_folder(split, label)

    print("--------------------------------")
    print(f"Kontrol sayfalari burada: {OUTPUT_DIR}")
    print("Islem tamamlandi.")


if __name__ == "__main__":
    main()