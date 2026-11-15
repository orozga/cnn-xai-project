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


def _compute_win_rates(
    results_by_method: dict[str, list[dict]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-metric win counts and win-rate percentages across shared images."""
    if len(results_by_method) < 2:
        return {}

    metric_directions = {
        "deletion_auc": False,
        "insertion_auc": True,
        "gini": True,
        "normalized_entropy": False,
        "compute_time_sec": False,
    }

    records_by_method: dict[str, dict[str, dict]] = {
        method: {r["image"]: r for r in records if "image" in r}
        for method, records in results_by_method.items()
    }
    common_images = sorted(
        set.intersection(*(set(recs.keys()) for recs in records_by_method.values()))
    )
    if not common_images:
        return {}

    win_counts: dict[str, dict[str, float]] = {
        method: {m: 0.0 for m in metric_directions} for method in results_by_method
    }

    for img_name in common_images:
        for metric_name, higher_is_better in metric_directions.items():
            values = {
                method: float(records_by_method[method][img_name][metric_name])
                for method in results_by_method
                if metric_name in records_by_method[method][img_name]
            }
            if not values:
                continue
            best_val = max(values.values()) if higher_is_better else min(values.values())
            winners = [
                method
                for method, val in values.items()
                if math.isclose(val, best_val, rel_tol=1e-9, abs_tol=1e-12)
            ]
            share = 1.0 / len(winners)
            for winner in winners:
                win_counts[winner][metric_name] += share

    total_images = len(common_images)
    return {
        method: {
            metric_name: {
                "wins": round(count, 2),
                "win_rate_pct": round(100.0 * count / total_images, 2),
            }
            for metric_name, count in metrics.items()
        }
        for method, metrics in win_counts.items()
    }


def save_comparison_report(
    summaries_by_method: dict[str, dict],
    output_dir: str | Path = "results/comparison",
    results_by_method: dict[str, list[dict]] | None = None,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    win_rates = _compute_win_rates(results_by_method) if results_by_method else {}
    enriched_summaries: dict[str, dict] = {}
    for method, summary in summaries_by_method.items():
        method_copy = {**summary}
        metrics_copy = {
            m_name: {**m_vals} for m_name, m_vals in summary.get("metrics", {}).items()
        }
        for m_name, wr in win_rates.get(method, {}).items():
            if m_name in metrics_copy:
                metrics_copy[m_name].update(wr)
        method_copy["metrics"] = metrics_copy
        enriched_summaries[method] = method_copy

    json_path = out_dir / "comparison_summary.json"
    json_path.write_text(
        json.dumps(enriched_summaries, indent=2, ensure_ascii=False) + "\n",
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

    if win_rates:
        lines.extend(
            [
                "",
                "## Procentowe zwycięstwa w poszczególnych metrykach (Win Rate %)",
                "",
                "| Metoda | Deletion AUC (↓) | Insertion AUC (↑) | Gini Index (↑) | Znorm. Entropia (↓) | Czas [s] (↓) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for method, method_wr in win_rates.items():
            lines.append(
                f"| **{method}** | "
                f"{method_wr['deletion_auc']['win_rate_pct']:.1f}% | "
                f"{method_wr['insertion_auc']['win_rate_pct']:.1f}% | "
                f"{method_wr['gini']['win_rate_pct']:.1f}% | "
                f"{method_wr['normalized_entropy']['win_rate_pct']:.1f}% | "
                f"{method_wr['compute_time_sec']['win_rate_pct']:.1f}% |"
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
