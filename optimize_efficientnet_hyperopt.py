import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset_processed"
OUTPUT_DIR = BASE_DIR / "model_outputs" / "efficientnet_b0_hyperopt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_EPOCHS = 3
SEED = 42
BATCH_SIZE_OPTIONS = [32, 64, 128]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms():
    return {
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


def build_datasets():
    transforms_dict = build_transforms()
    return {
        split: datasets.ImageFolder(DATA_DIR / split, transform=transforms_dict[split])
        for split in ["train", "val", "test"]
    }


def build_loaders(datasets_dict, batch_size):
    return {
        split: DataLoader(
            datasets_dict[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0
        )
        for split in ["train", "val", "test"]
    }


def build_model(num_classes, device):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)


def run_epoch(model, loader, dataset_size, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    running_loss = 0.0
    running_correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += torch.sum(preds == labels).item()

    return running_loss / dataset_size, running_correct / dataset_size


def train_trial(params, datasets_dict, class_names, device, trial_number):
    batch_size = int(params["batch_size"])
    lr = float(params["lr"])
    weight_decay = float(params["weight_decay"])

    set_seed(SEED + trial_number)
    loaders = build_loaders(datasets_dict, batch_size)
    sizes = {split: len(datasets_dict[split]) for split in ["train", "val", "test"]}
    model = build_model(len(class_names), device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    history = []
    start = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = run_epoch(
            model, loaders["train"], sizes["train"], criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, loaders["val"], sizes["val"], criterion, optimizer, device, train=False
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                OUTPUT_DIR / f"trial_{trial_number:02d}_best_model.pth"
            )

    elapsed = (time.time() - start) / 60

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "loss": 1.0 - best_val_acc,
        "status": STATUS_OK,
        "trial_number": trial_number,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "elapsed_minutes": elapsed,
        "history": history,
        "model_path": str(OUTPUT_DIR / f"trial_{trial_number:02d}_best_model.pth")
    }


def evaluate_best_model(best_params, datasets_dict, class_names, device):
    batch_size = int(best_params["batch_size"])
    loaders = build_loaders(datasets_dict, batch_size)
    model = build_model(len(class_names), device)
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_efficientnet_b0_hyperopt_model.pth"))
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

    with open(OUTPUT_DIR / "test_report.txt", "w", encoding="utf-8") as f:
        f.write("EfficientNet-B0 Hyperopt Test Report\n")
        f.write("====================================\n\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(str(cm))

    plt.figure()
    plt.imshow(cm)
    plt.title("EfficientNet-B0 Hyperopt Confusion Matrix")
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

    return test_acc, cm, report


def save_trial_results(results):
    csv_path = OUTPUT_DIR / "hyperopt_trials.csv"
    fieldnames = [
        "trial_number",
        "lr",
        "weight_decay",
        "batch_size",
        "best_val_acc",
        "best_val_loss",
        "elapsed_minutes"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-evals", type=int, default=10)
    args = parser.parse_args()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"dataset_processed bulunamadi: {DATA_DIR}")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_dict = build_datasets()
    class_names = datasets_dict["train"].classes

    print("EfficientNet-B0 Hyperopt optimizasyonu basliyor...")
    print("--------------------------------")
    print("Cihaz:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Siniflar:", class_names)
    print("Arama alani:")
    print("learning_rate: 1e-5 - 1e-2")
    print("weight_decay: 1e-6 - 1e-3")
    print("batch_size:", BATCH_SIZE_OPTIONS)
    print("optimizer: Adam")
    print("max_evals:", args.max_evals)

    space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-2)),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-3)),
        "batch_size": hp.choice("batch_size", BATCH_SIZE_OPTIONS)
    }

    trials = Trials()
    results = []
    global_best = {"val_acc": -1.0, "val_loss": float("inf"), "model_path": None}

    def objective(params):
        trial_number = len(results) + 1
        print()
        print(f"Trial {trial_number}/{args.max_evals}")
        print(
            "lr={:.8f}, weight_decay={:.8f}, batch_size={}".format(
                params["lr"], params["weight_decay"], params["batch_size"]
            )
        )
        result = train_trial(params, datasets_dict, class_names, device, trial_number)
        results.append(result)
        if (
            result["best_val_acc"] > global_best["val_acc"]
            or (
                result["best_val_acc"] == global_best["val_acc"]
                and result["best_val_loss"] < global_best["val_loss"]
            )
        ):
            global_best["val_acc"] = result["best_val_acc"]
            global_best["val_loss"] = result["best_val_loss"]
            global_best["model_path"] = result["model_path"]
            shutil.copyfile(
                result["model_path"],
                OUTPUT_DIR / "best_efficientnet_b0_hyperopt_model.pth"
            )
            print("Genel en iyi model guncellendi.")
        print(
            "best_val_acc={:.4f}, best_val_loss={:.4f}, sure={:.2f} dk".format(
                result["best_val_acc"], result["best_val_loss"], result["elapsed_minutes"]
            )
        )
        return result

    best_raw = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
        rstate=np.random.default_rng(SEED)
    )

    best_params = space_eval(space, best_raw)
    best_result = max(results, key=lambda item: (item["best_val_acc"], -item["best_val_loss"]))

    save_trial_results(results)

    with open(OUTPUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": best_params,
                "best_trial": best_result,
                "search_space": {
                    "learning_rate": "1e-5 - 1e-2",
                    "weight_decay": "1e-6 - 1e-3",
                    "batch_size": BATCH_SIZE_OPTIONS,
                    "optimizer": "Adam"
                }
            },
            f,
            indent=2
        )

    test_acc, cm, report = evaluate_best_model(best_params, datasets_dict, class_names, device)

    print()
    print("OPTIMIZASYON TAMAMLANDI")
    print("--------------------------------")
    print("Best params:", best_params)
    print("Best validation accuracy:", round(best_result["best_val_acc"], 4))
    print("Test accuracy:", round(test_acc, 4))
    print(report)
    print("Confusion matrix:")
    print(cm)
    print("Ciktilar:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
