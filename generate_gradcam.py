import csv
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset_processed"
OUTPUT_DIR = BASE_DIR / "model_outputs" / "explainable_ai_gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES_PER_CLASS = 3


PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def build_model(model_key, num_classes):
    if model_key == "resnet50_baseline":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        checkpoint = BASE_DIR / "model_outputs" / "resnet50_baseline" / "best_resnet50_model.pth"
        target_layer = model.layer4[-1]
    elif model_key == "efficientnet_b0_hyperopt":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        checkpoint = BASE_DIR / "model_outputs" / "efficientnet_b0_hyperopt" / "best_efficientnet_b0_hyperopt_model.pth"
        target_layer = model.features[-1]
    elif model_key == "efficientnet_b0_optuna":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        checkpoint = BASE_DIR / "model_outputs" / "efficientnet_b0_optuna" / "best_efficientnet_b0_optuna_model.pth"
        target_layer = model.features[-1]
    else:
        raise ValueError(f"Bilinmeyen model: {model_key}")

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, target_layer, checkpoint


class CamExtractor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self.activations = output

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor, target_class, method):
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        score = output[:, target_class].sum()
        score.backward(retain_graph=True)

        activations = self.activations.detach()
        gradients = self.gradients.detach()

        if method == "gradcam":
            weights = gradients.mean(dim=(2, 3), keepdim=True)
        elif method == "gradcampp":
            gradients_power_2 = gradients.pow(2)
            gradients_power_3 = gradients.pow(3)
            sum_activations = activations.sum(dim=(2, 3), keepdim=True)
            denominator = 2 * gradients_power_2 + sum_activations * gradients_power_3
            denominator = torch.where(
                denominator != 0,
                denominator,
                torch.ones_like(denominator)
            )
            alpha = gradients_power_2 / denominator
            positive_gradients = torch.relu(gradients)
            weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)
        else:
            raise ValueError(f"Bilinmeyen CAM metodu: {method}")

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, output.detach()


def select_samples(dataset):
    selected = []
    counts = {class_name: 0 for class_name in dataset.classes}

    for path, label_idx in dataset.samples:
        class_name = dataset.classes[label_idx]
        if counts[class_name] < NUM_SAMPLES_PER_CLASS:
            selected.append((Path(path), label_idx, class_name))
            counts[class_name] += 1
        if all(count >= NUM_SAMPLES_PER_CLASS for count in counts.values()):
            break

    return selected


def make_overlay(image_array, cam, alpha=0.45):
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[..., :3]
    overlay = (1 - alpha) * image_array + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return heatmap, overlay


def save_visuals(output_prefix, image_array, heatmap, overlay, title):
    original_path = output_prefix.with_name(output_prefix.name + "_original.png")
    heatmap_path = output_prefix.with_name(output_prefix.name + "_heatmap.png")
    overlay_path = output_prefix.with_name(output_prefix.name + "_overlay.png")
    panel_path = output_prefix.with_name(output_prefix.name + "_panel.png")

    plt.imsave(original_path, image_array)
    plt.imsave(heatmap_path, heatmap)
    plt.imsave(overlay_path, overlay)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(image_array)
    axes[0].set_title("Original")
    axes[1].imshow(heatmap)
    axes[1].set_title("Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return original_path, heatmap_path, overlay_path, panel_path


def process_model(model_key, dataset, samples):
    model, target_layer, checkpoint = build_model(model_key, len(dataset.classes))
    extractor = CamExtractor(model, target_layer)
    model_dir = OUTPUT_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    try:
        for image_path, true_idx, true_class in samples:
            image = Image.open(image_path).convert("RGB")
            image_array = np.asarray(image).astype(np.float32) / 255.0
            input_tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_idx = int(torch.argmax(probabilities, dim=1).item())
                confidence = float(probabilities[0, predicted_idx].item())

            predicted_class = dataset.classes[predicted_idx]
            is_correct = predicted_idx == true_idx

            for method in ["gradcam", "gradcampp"]:
                cam, _ = extractor.generate(input_tensor, predicted_idx, method)
                heatmap, overlay = make_overlay(image_array, cam)
                stem = f"{true_class}_{image_path.stem}_{method}"
                output_prefix = model_dir / stem
                original_path, heatmap_path, overlay_path, panel_path = save_visuals(
                    output_prefix,
                    image_array,
                    heatmap,
                    overlay,
                    f"{model_key} | {method} | true={true_class} pred={predicted_class}"
                )

                rows.append({
                    "model": model_key,
                    "method": method,
                    "image": str(image_path),
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "confidence": f"{confidence:.6f}",
                    "correct": is_correct,
                    "original": str(original_path),
                    "heatmap": str(heatmap_path),
                    "overlay": str(overlay_path),
                    "panel": str(panel_path),
                    "checkpoint": str(checkpoint)
                })
    finally:
        extractor.close()

    return rows


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"dataset_processed bulunamadi: {DATA_DIR}")

    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=None)
    samples = select_samples(test_dataset)
    model_keys = [
        "resnet50_baseline",
        "efficientnet_b0_hyperopt",
        "efficientnet_b0_optuna"
    ]

    print("Grad-CAM / Grad-CAM++ analizi basliyor...")
    print("--------------------------------")
    print("Cihaz:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Siniflar:", test_dataset.classes)
    print("Secilen ornek sayisi:", len(samples))

    all_rows = []
    for model_key in model_keys:
        print()
        print("Model:", model_key)
        rows = process_model(model_key, test_dataset, samples)
        all_rows.extend(rows)
        correct_count = sum(1 for row in rows if row["method"] == "gradcam" and row["correct"])
        print(f"Dogru tahmin edilen secili ornekler: {correct_count}/{len(samples)}")

    summary_path = OUTPUT_DIR / "gradcam_summary.csv"
    fieldnames = [
        "model",
        "method",
        "image",
        "true_class",
        "predicted_class",
        "confidence",
        "correct",
        "original",
        "heatmap",
        "overlay",
        "panel",
        "checkpoint"
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    report_path = OUTPUT_DIR / "gradcam_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Grad-CAM / Grad-CAM++ Report\n")
        f.write("============================\n\n")
        f.write(f"Models: {model_keys}\n")
        f.write(f"Classes: {test_dataset.classes}\n")
        f.write(f"Samples per class: {NUM_SAMPLES_PER_CLASS}\n")
        f.write(f"Total selected samples: {len(samples)}\n")
        f.write(f"Methods: Grad-CAM, Grad-CAM++\n\n")
        f.write("Selected samples:\n")
        for image_path, _true_idx, true_class in samples:
            f.write(f"- {true_class}: {image_path}\n")
        f.write("\nSummary CSV:\n")
        f.write(str(summary_path))

    print()
    print("Grad-CAM / Grad-CAM++ analizi tamamlandi.")
    print("Ciktilar:", OUTPUT_DIR)
    print("Ozet:", summary_path)


if __name__ == "__main__":
    main()
