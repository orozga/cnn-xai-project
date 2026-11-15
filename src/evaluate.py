from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.metrics import (
    compute_deletion_curve,
    compute_insertion_curve,
    compute_normalized_entropy,
    compute_sparsity_gini,
    randomize_layers_cascading,
)
from src.utils import (
    get_imagenet_mapping,
    get_sample_images_paths,
    load_and_preprocess_image,
    load_sample_metadata,
    save_comparison_report,
    save_metrics_results,
    save_metrics_summary,
)
from src.xai_methods import (
    AVAILABLE_METHODS,
    METHOD_LABELS,
    explain_image,
    plot_methods_comparison,
    plot_saliency_map,
)


def run_full_evaluation(
    model_wrapper,
    image_dir: str | Path = "data/sample_images",
    output_path: str | Path | None = None,
    steps: int = 20,
    method: str = "saliency",
    max_images: int | None = None,
    scorecam_max_channels: int | None = None,
) -> list[dict]:
    method_key = method.lower().strip()
    if output_path is None:
        output_path = f"results/{method_key}/metrics_results.json"

    paths = sorted(get_sample_images_paths(str(image_dir)))
    if not paths:
        raise FileNotFoundError(
            f"No sample images found in {Path(image_dir).resolve()}"
        )
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]

    model = model_wrapper.model
    device = model_wrapper.device
    class_mapping = get_imagenet_mapping()
    sample_metadata = load_sample_metadata(image_dir)
    results = []
    done: dict[str, int] = {}

    for image_path in paths:
        original_image, input_batch = load_and_preprocess_image(image_path)
        input_batch = input_batch.to(device)

        explanation = explain_image(
            model_wrapper,
            input_batch,
            method=method_key,
            max_channels=scorecam_max_channels,
        )
        heatmap = explanation["heatmap"]
        class_idx = explanation["target_class_idx"]
        compute_time_sec = explanation["compute_time_sec"]

        _, _, deletion_auc = compute_deletion_curve(
            model, input_batch, heatmap, class_idx, steps=steps, device=device
        )
        _, _, insertion_auc = compute_insertion_curve(
            model, input_batch, heatmap, class_idx, steps=steps, device=device
        )
        gini = compute_sparsity_gini(heatmap)
        entropy = compute_normalized_entropy(heatmap)

        class_name = class_mapping.get(class_idx, "Unknown")
        done[class_name] = done.get(class_name, 0) + 1

        label = METHOD_LABELS.get(method_key, method_key)
        output_image = (
            Path("results")
            / method_key
            / f"{method_key}_{class_name}_{done[class_name]}.png"
        )
        output_image.parent.mkdir(parents=True, exist_ok=True)
        plot_saliency_map(
            original_image,
            heatmap,
            title=(
                f"{label}: {class_name} ({class_idx}) | "
                f"Gini: {gini:.3f} | Entropy: {entropy:.3f}"
            ),
            save_path=str(output_image),
        )

        meta_entry = sample_metadata.get(image_path.name)
        gt_idx = meta_entry["ground_truth_idx"] if meta_entry else None
        is_correct = (int(class_idx) == int(gt_idx)) if gt_idx is not None else None

        results.append(
            {
                "image": image_path.name,
                "method": method_key,
                "target_class_idx": int(class_idx),
                "target_class_name": class_name,
                "ground_truth_idx": gt_idx,
                "is_correct": is_correct,
                "deletion_auc": deletion_auc,
                "insertion_auc": insertion_auc,
                "gini": gini,
                "normalized_entropy": entropy,
                "compute_time_sec": compute_time_sec,
            }
        )

    save_metrics_results(results, output_path)
    return results


def run_comparative_pipeline(
    model_wrapper,
    methods: Sequence[str] = AVAILABLE_METHODS,
    image_dir: str | Path = "data/sample_images",
    steps: int = 20,
    max_images: int | None = None,
    scorecam_max_channels: int | None = None,
) -> dict[str, dict]:
    """Run unified XAI evaluation across all specified methods and generate comparison reports."""
    paths = sorted(get_sample_images_paths(str(image_dir)))
    if not paths:
        raise FileNotFoundError(
            f"No sample images found in {Path(image_dir).resolve()}"
        )
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]

    model = model_wrapper.model
    device = model_wrapper.device
    class_mapping = get_imagenet_mapping()
    sample_metadata = load_sample_metadata(image_dir)

    results_by_method: dict[str, list[dict]] = {m: [] for m in methods}
    done_per_method: dict[str, dict[str, int]] = {m: {} for m in methods}
    done_comparison: dict[str, int] = {}

    comparison_dir = Path("results/comparison")
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for image_path in paths:
        original_image, input_batch = load_and_preprocess_image(image_path)
        input_batch = input_batch.to(device)

        with np.errstate(all="ignore"):
            logits = model_wrapper.predict(input_batch)
            target_class_idx = int(logits.argmax(dim=1).item())

        class_name = class_mapping.get(target_class_idx, "Unknown")
        meta_entry = sample_metadata.get(image_path.name)
        gt_idx = meta_entry["ground_truth_idx"] if meta_entry else None
        is_correct = (
            (int(target_class_idx) == int(gt_idx)) if gt_idx is not None else None
        )

        per_image_explanations: dict[str, dict] = {}

        for method_key in methods:
            explanation = explain_image(
                model_wrapper,
                input_batch,
                method=method_key,
                target_class_idx=target_class_idx,
                max_channels=scorecam_max_channels,
            )
            heatmap = explanation["heatmap"]
            compute_time_sec = explanation["compute_time_sec"]

            _, _, deletion_auc = compute_deletion_curve(
                model,
                input_batch,
                heatmap,
                target_class_idx,
                steps=steps,
                device=device,
            )
            _, _, insertion_auc = compute_insertion_curve(
                model,
                input_batch,
                heatmap,
                target_class_idx,
                steps=steps,
                device=device,
            )
            gini = compute_sparsity_gini(heatmap)
            entropy = compute_normalized_entropy(heatmap)

            done_per_method[method_key][class_name] = (
                done_per_method[method_key].get(class_name, 0) + 1
            )
            idx_for_class = done_per_method[method_key][class_name]

            label = METHOD_LABELS.get(method_key, method_key)
            output_image = (
                Path("results")
                / method_key
                / f"{method_key}_{class_name}_{idx_for_class}.png"
            )
            output_image.parent.mkdir(parents=True, exist_ok=True)
            plot_saliency_map(
                original_image,
                heatmap,
                title=(
                    f"{label}: {class_name} ({target_class_idx}) | "
                    f"Gini: {gini:.3f} | Entropy: {entropy:.3f}"
                ),
                save_path=str(output_image),
            )

            record = {
                "image": image_path.name,
                "method": method_key,
                "target_class_idx": target_class_idx,
                "target_class_name": class_name,
                "ground_truth_idx": gt_idx,
                "is_correct": is_correct,
                "deletion_auc": deletion_auc,
                "insertion_auc": insertion_auc,
                "gini": gini,
                "normalized_entropy": entropy,
                "compute_time_sec": compute_time_sec,
            }
            results_by_method[method_key].append(record)
            per_image_explanations[method_key] = {
                "heatmap": heatmap,
                "deletion_auc": deletion_auc,
                "insertion_auc": insertion_auc,
                "gini": gini,
                "normalized_entropy": entropy,
                "compute_time_sec": compute_time_sec,
            }

        done_comparison[class_name] = done_comparison.get(class_name, 0) + 1
        comp_path = (
            comparison_dir
            / f"comparison_{class_name}_{done_comparison[class_name]}.png"
        )
        plot_methods_comparison(
            original_image,
            per_image_explanations,
            class_name=class_name,
            class_idx=target_class_idx,
            save_path=str(comp_path),
        )

    summaries_by_method: dict[str, dict] = {}
    for method_key, method_results in results_by_method.items():
        save_metrics_results(
            method_results, f"results/{method_key}/metrics_results.json"
        )
        summary = save_metrics_summary(
            method_results, f"results/{method_key}/metrics_summary.json"
        )
        summaries_by_method[method_key] = summary

    save_comparison_report(summaries_by_method, output_dir=comparison_dir)
    return summaries_by_method


def run_sanity_check(
    image_path: str | Path,
    output_path: str | Path,
    method: str = "saliency",
    scorecam_max_channels: int | None = 128,
) -> None:
    from src.model import XAIResNetModel

    model_wrapper = XAIResNetModel()
    model = model_wrapper.model
    device = model_wrapper.device

    original_image, input_batch = load_and_preprocess_image(image_path)
    input_batch = input_batch.to(device)

    layer_groups = [
        ("Baseline (Original weights)", []),
        ("Randomized classifier", ["fc"]),
        ("Randomized layer4", ["layer4"]),
        ("Randomized layer3", ["layer3"]),
    ]

    base_expl = explain_image(
        model_wrapper,
        input_batch,
        method=method,
        max_channels=scorecam_max_channels,
    )
    class_idx = base_expl["target_class_idx"]
    heatmaps = [base_expl["heatmap"]]

    for _, prefixes in layer_groups[1:]:
        layer_names = [
            name
            for name, _ in model.named_modules()
            if any(name.startswith(prefix) for prefix in prefixes)
        ]

        randomize_layers_cascading(model, layer_names)

        expl = explain_image(
            model_wrapper,
            input_batch,
            method=method,
            target_class_idx=class_idx,
            max_channels=scorecam_max_channels,
        )
        heatmaps.append(expl["heatmap"])

    figure, axes = plt.subplots(1, len(layer_groups) + 1, figsize=(25, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    orig_width, orig_height = original_image.size

    for axis, (title, _), saliency in zip(axes[1:], layer_groups, heatmaps):
        saliency_pil = Image.fromarray((255 * saliency).astype(np.uint8))
        saliency_resized = saliency_pil.resize(
            (orig_width, orig_height), Image.Resampling.BILINEAR
        )
        saliency_resized_np = np.array(saliency_resized) / 255.0

        axis.imshow(saliency_resized_np, cmap="hot", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=300)
    plt.close(figure)

    model_wrapper.remove_hooks()
    print(f"Saved Adebayo test results ({method}) in: {destination}")


def run_saliency_sanity_check(
    image_path: str | Path,
    output_path: str | Path,
) -> None:
    run_sanity_check(image_path, output_path, method="saliency")
