import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset_processed"
OUTPUT_DIR = BASE_DIR / "model_outputs" / "resnet50_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 3
LR = 0.0001
WEIGHT_DECAY = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("ResNet50 baseline egitimi basliyor...")
print("--------------------------------")
print("Cihaz:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if not DATA_DIR.exists():
    raise FileNotFoundError(f"dataset_processed bulunamadi: {DATA_DIR}")


transforms_dict = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


datasets_dict = {
    split: datasets.ImageFolder(DATA_DIR / split, transform=transforms_dict[split])
    for split in ["train", "val", "test"]
}

loaders = {
    split: DataLoader(
        datasets_dict[split],
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=0
    )
    for split in ["train", "val", "test"]
}

sizes = {split: len(datasets_dict[split]) for split in ["train", "val", "test"]}
class_names = datasets_dict["train"].classes

print("Siniflar:", class_names)
print("Veri sayilari:")
for split in ["train", "val", "test"]:
    print(split, sizes[split])


model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_acc = 0.0
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

start = time.time()

for epoch in range(NUM_EPOCHS):
    print()
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("--------------------------------")

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_correct = 0

        for images, labels in loaders[phase]:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += torch.sum(preds == labels).item()

        epoch_loss = running_loss / sizes[phase]
        epoch_acc = running_correct / sizes[phase]

        print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

        history[f"{phase}_loss"].append(epoch_loss)
        history[f"{phase}_acc"].append(epoch_acc)

        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_resnet50_model.pth")
            print("En iyi model kaydedildi.")


elapsed = (time.time() - start) / 60
print()
print(f"Egitim tamamlandi. Sure: {elapsed:.2f} dakika")
print(f"En iyi validation accuracy: {best_acc:.4f}")


model.load_state_dict(torch.load(OUTPUT_DIR / "best_resnet50_model.pth"))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in loaders["test"]:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

test_acc = np.mean(np.array(all_labels) == np.array(all_preds))
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=4
)

print()
print("TEST SONUCLARI")
print("--------------------------------")
print("Test accuracy:", round(test_acc, 4))
print(report)
print("Confusion matrix:")
print(cm)


with open(OUTPUT_DIR / "test_report.txt", "w", encoding="utf-8") as f:
    f.write("ResNet50 Baseline Test Report\n")
    f.write("=============================\n\n")
    f.write(f"Classes: {class_names}\n")
    f.write(f"Best val accuracy: {best_acc:.4f}\n")
    f.write(f"Test accuracy: {test_acc:.4f}\n\n")
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(str(cm))


epochs = range(1, NUM_EPOCHS + 1)

plt.figure()
plt.plot(epochs, history["train_loss"], label="train_loss")
plt.plot(epochs, history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("ResNet50 Loss")
plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(epochs, history["train_acc"], label="train_acc")
plt.plot(epochs, history["val_acc"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("ResNet50 Accuracy")
plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(len(class_names)), class_names)
plt.yticks(range(len(class_names)), class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print()
print("--------------------------------")
print("Tum ciktilar kaydedildi:")
print(OUTPUT_DIR)