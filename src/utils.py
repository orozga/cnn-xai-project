import json
import math
import urllib.request
from pathlib import Path
from typing import cast

from PIL import Image
from torch import Tensor
from torchvision import transforms


def load_and_preprocess_image(image_path: str | Path) -> tuple[Image.Image, Tensor]:
    crop_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )
    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    cropped_img = cast(Image.Image, crop_transform(img))
    input_tensor = cast(Tensor, tensor_transform(cropped_img))
    input_batch = input_tensor.unsqueeze(0)

    return cropped_img, input_batch


def get_sample_images_paths(folder_path: str = "data/sample_images") -> list[Path]:
    folder = Path(folder_path)
    if not folder.exists():
        return []
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in valid_extensions]
    )


def load_sample_metadata(folder_path: str | Path = "data/sample_images") -> dict:
    metadata_path = Path(folder_path) / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def save_saliency_metrics(
    results: list[dict], output_path: str | Path
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


save_metrics_results = save_saliency_metrics


def save_metrics_summary(
    results: list[dict], output_path: str | Path
) -> dict:
    metric_names = (
        "deletion_auc",
        "insertion_auc",
        "gini",
        "normalized_entropy",
        "compute_time_sec",
    )
    summary: dict = {
        "image_count": len(results),
        "metrics": {},
    }

    correct_flags = [
        bool(result["is_correct"])
        for result in results
        if "is_correct" in result and result["is_correct"] is not None
    ]
    if correct_flags:
        summary["top1_accuracy"] = round(
            sum(correct_flags) / len(correct_flags), 6
        )

    for metric_name in metric_names:
        values = [
            float(result[metric_name])
            for result in results
            if metric_name in result and result[metric_name] is not None
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
    return summary


def save_comparison_report(
    summaries_by_method: dict[str, dict],
    output_dir: str | Path = "results/comparison",
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "comparison_summary.json"
    json_path.write_text(
        json.dumps(summaries_by_method, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Raport Porównawczy Metod XAI (Faza I i Faza II)",
        "",
        "| Metoda | Liczba obrazów | Deletion AUC (↓) | Insertion AUC (↑) | Gini Index (↑) | Znorm. Entropia (↓) | Śr. czas [s] |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    def _fmt_metric(metrics_dict: dict, m_name: str) -> str:
        entry = metrics_dict.get(m_name)
        if not entry:
            return "-"
        return f"{entry['mean']:.4f} ± {entry['std']:.4f}"

    for method, summary in summaries_by_method.items():
        count = summary.get("image_count", 0)
        metrics = summary.get("metrics", {})

        lines.append(
            f"| **{method}** | {count} | {_fmt_metric(metrics, 'deletion_auc')} | "
            f"{_fmt_metric(metrics, 'insertion_auc')} | {_fmt_metric(metrics, 'gini')} | "
            f"{_fmt_metric(metrics, 'normalized_entropy')} | {_fmt_metric(metrics, 'compute_time_sec')} |"
        )

    acc_values = [
        s["top1_accuracy"]
        for s in summaries_by_method.values()
        if "top1_accuracy" in s
    ]
    if acc_values:
        lines.extend(
            [
                "",
                f"**Skuteczność klasyfikatora bazowego ResNet-50 (Top-1 Accuracy):** `{acc_values[0] * 100:.2f}%`",
            ]
        )

    md_path = out_dir / "RAPORT_FAZA_2.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
