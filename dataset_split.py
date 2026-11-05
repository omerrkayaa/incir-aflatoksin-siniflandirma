import os
import shutil
import random
from pathlib import Path

# Bu kod hangi klasördeyse orayı ana proje klasörü kabul eder.
BASE_DIR = Path(__file__).resolve().parent

HEALTHY_1 = BASE_DIR / "Healty"
HEALTHY_2 = BASE_DIR / "mor sağlıklı" / "Mor sağlıklı"
UNHEALTHY_ORIGINAL = BASE_DIR / "aflatoksin son" / "Aflatoksin yeni"
AUGMENTED = BASE_DIR / "augmented"

OUTPUT = BASE_DIR / "dataset_split"

random.seed(42)

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def check_folder(folder):
    if not folder.exists():
        print(f"HATA: Klasör bulunamadı: {folder}", flush=True)
        return False
    print(f"Klasör bulundu: {folder}", flush=True)
    return True


def get_images(folder):
    return [
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMG_EXTENSIONS
    ]


def split_data(files, train_ratio=0.70, val_ratio=0.15):
    files = files.copy()
    random.shuffle(files)

    total = len(files)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    return train_files, val_files, test_files


def unique_destination_path(target_folder, file_name):
    destination = target_folder / file_name

    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 1

    while True:
        new_destination = target_folder / f"{stem}_copy{counter}{suffix}"
        if not new_destination.exists():
            return new_destination
        counter += 1


def add_files_as_hardlink(files, target_folder, title):
    """
    Dosyaları kopyalamaz.
    Hard link oluşturur.
    Yani disk alanını çok az kullanır.
    """
    target_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{title} ekleniyor. Toplam dosya: {len(files)}", flush=True)

    for i, file in enumerate(files, start=1):
        destination = unique_destination_path(target_folder, file.name)

        try:
            os.link(file, destination)
        except OSError as e:
            print("\nHATA: Hard link oluşturulamadı.", flush=True)
            print(f"Kaynak dosya: {file}", flush=True)
            print(f"Hedef dosya: {destination}", flush=True)
            print(f"Hata mesajı: {e}", flush=True)
            print("\nBu hata tekrar olursa CSV yöntemiyle devam edeceğiz.", flush=True)
            raise

        if i % 50 == 0 or i == len(files):
            print(f"{title}: {i}/{len(files)} dosya eklendi.", flush=True)


print("Program başladı.", flush=True)
print("--------------------------------", flush=True)

folders_ok = True
folders_ok &= check_folder(HEALTHY_1)
folders_ok &= check_folder(HEALTHY_2)
folders_ok &= check_folder(UNHEALTHY_ORIGINAL)
folders_ok &= check_folder(AUGMENTED)

if not folders_ok:
    print("\nKlasör yollarında hata var. Program durduruldu.", flush=True)
    exit()

print("\nGörüntüler okunuyor...", flush=True)

healthy_files = get_images(HEALTHY_1) + get_images(HEALTHY_2)
unhealthy_original_files = get_images(UNHEALTHY_ORIGINAL)
augmented_files = get_images(AUGMENTED)

print(f"Healthy toplam: {len(healthy_files)}", flush=True)
print(f"Unhealthy orijinal toplam: {len(unhealthy_original_files)}", flush=True)
print(f"Augmented toplam: {len(augmented_files)}", flush=True)

if OUTPUT.exists():
    print(f"\nEski çıktı klasörü siliniyor: {OUTPUT}", flush=True)
    shutil.rmtree(OUTPUT)

print("\nOrijinal veriler train / val / test olarak ayrılıyor...", flush=True)

healthy_train, healthy_val, healthy_test = split_data(healthy_files)
unhealthy_train, unhealthy_val, unhealthy_test = split_data(unhealthy_original_files)

train_unhealthy_stems = {file.stem for file in unhealthy_train}

selected_augmented = []

for aug_file in augmented_files:
    base_name = aug_file.stem.split("_aug")[0]

    if base_name in train_unhealthy_stems:
        selected_augmented.append(aug_file)

print("\nBölme sonucu:", flush=True)
print("--------------------------------", flush=True)
print(f"Train Healthy: {len(healthy_train)}", flush=True)
print(f"Val Healthy: {len(healthy_val)}", flush=True)
print(f"Test Healthy: {len(healthy_test)}", flush=True)
print()
print(f"Train Unhealthy Original: {len(unhealthy_train)}", flush=True)
print(f"Train Unhealthy Augmented: {len(selected_augmented)}", flush=True)
print(f"Val Unhealthy: {len(unhealthy_val)}", flush=True)
print(f"Test Unhealthy: {len(unhealthy_test)}", flush=True)
print("--------------------------------", flush=True)

print("\nDosyalar dataset_split klasörüne ekleniyor...", flush=True)

add_files_as_hardlink(
    healthy_train,
    OUTPUT / "train" / "healthy",
    "Train Healthy"
)

add_files_as_hardlink(
    healthy_val,
    OUTPUT / "val" / "healthy",
    "Val Healthy"
)

add_files_as_hardlink(
    healthy_test,
    OUTPUT / "test" / "healthy",
    "Test Healthy"
)

add_files_as_hardlink(
    unhealthy_train,
    OUTPUT / "train" / "unhealthy",
    "Train Unhealthy Original"
)

add_files_as_hardlink(
    unhealthy_val,
    OUTPUT / "val" / "unhealthy",
    "Val Unhealthy"
)

add_files_as_hardlink(
    unhealthy_test,
    OUTPUT / "test" / "unhealthy",
    "Test Unhealthy"
)

add_files_as_hardlink(
    selected_augmented,
    OUTPUT / "train" / "unhealthy",
    "Train Unhealthy Augmented"
)

report_path = OUTPUT / "split_report.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("Dataset Split Raporu\n")
    report.write("====================\n\n")

    report.write(f"Healthy toplam: {len(healthy_files)}\n")
    report.write(f"Unhealthy orijinal toplam: {len(unhealthy_original_files)}\n")
    report.write(f"Augmented toplam: {len(augmented_files)}\n\n")

    report.write(f"Train Healthy: {len(healthy_train)}\n")
    report.write(f"Val Healthy: {len(healthy_val)}\n")
    report.write(f"Test Healthy: {len(healthy_test)}\n\n")

    report.write(f"Train Unhealthy Original: {len(unhealthy_train)}\n")
    report.write(f"Train Unhealthy Augmented: {len(selected_augmented)}\n")
    report.write(f"Val Unhealthy: {len(unhealthy_val)}\n")
    report.write(f"Test Unhealthy: {len(unhealthy_test)}\n\n")

    report.write("Not:\n")
    report.write("Augmented görüntüler yalnızca train/unhealthy klasörüne eklenmiştir.\n")
    report.write("Validation ve test klasörlerinde augmented görüntü kullanılmamıştır.\n")
    report.write("Bu işlem data leakage riskini azaltmak için yapılmıştır.\n")
    report.write("Dosyalar hard link olarak eklendiği için disk alanı gereksiz yere artmamıştır.\n")

print("\n--------------------------------", flush=True)
print("Dataset bölme işlemi tamamlandı.", flush=True)
print(f"Çıktı klasörü: {OUTPUT}", flush=True)
print(f"Rapor dosyası: {report_path}", flush=True)
print("--------------------------------", flush=True)