import json
import math
from pathlib import Path
from PIL import Image
from typing import Tuple, cast
from torch import Tensor
from torchvision import transforms
import torch
import urllib.request


def load_and_preprocess_image(image_path: str | Path) -> Tuple[Image.Image, Tensor]:
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    input_tensor = cast(Tensor, preprocess(img))
    input_batch = input_tensor.unsqueeze(0)

    return img, input_batch

def get_sample_images_paths(folder_path: str = "data/sample_images") -> list[Path]:
    folder = Path(folder_path)
    if not folder.exists():
        return []
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in folder.iterdir() if p.suffix.lower() in valid_extensions]


def save_saliency_metrics(
    results: list[dict], output_path: str | Path
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_metrics_summary(
    results: list[dict], output_path: str | Path
) -> None:
    metric_names = (
        "deletion_auc",
        "insertion_auc",
        "gini",
        "normalized_entropy",
    )
    summary = {
        "image_count": len(results),
        "metrics": {},
    }

    for metric_name in metric_names:
        values = [
            float(result[metric_name])
            for result in results
            if metric_name in result
        ]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summary["metrics"][metric_name] = {
            "mean": round(mean, 6),
            "std": round(math.sqrt(variance), 6),
            "minimum": round(min(values), 6),
            "maximum": round(max(values), 6),
        }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def get_imagenet_mapping(output_path: str = "data/imagenet_class_index.json") -> dict:
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    file_path = Path(output_path)

    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading ImageNet class mapping...")
        urllib.request.urlretrieve(url, file_path)

    with open(file_path, "r") as f:
        raw_mapping = json.load(f)

    idx_to_label = {int(idx): info[1] for idx, info in raw_mapping.items()}

    return idx_to_label
