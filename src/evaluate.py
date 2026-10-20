import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
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
    save_saliency_metrics,
)
from src.xai_methods import compute_saliency_map, plot_saliency_map


def run_full_evaluation(
    model_wrapper,
    image_dir: str | Path = "data/sample_images",
    output_path: str | Path = "results/saliency/metrics_results.json",
    steps: int = 20,
) -> list[dict]:
    paths = sorted(get_sample_images_paths(str(image_dir)))
    if not paths:
        raise FileNotFoundError(
            f"No sample images found in {Path(image_dir).resolve()}"
        )

    model = model_wrapper.model
    device = model_wrapper.device
    class_mapping = get_imagenet_mapping()
    results = []
    done = {}

    for image_path in paths:
        original_image, input_batch = load_and_preprocess_image(image_path)
        input_batch = input_batch.to(device)
        saliency, class_idx = compute_saliency_map(model, input_batch)

        _, _, deletion_auc = compute_deletion_curve(
            model, input_batch, saliency, class_idx, steps=steps, device=device
        )
        _, _, insertion_auc = compute_insertion_curve(
            model, input_batch, saliency, class_idx, steps=steps, device=device
        )
        gini = compute_sparsity_gini(saliency)
        entropy = compute_normalized_entropy(saliency)

        class_name = class_mapping.get(class_idx, "Unknown")
        if class_name not in done:
            done[class_name] = 1
        else:
            done[class_name] += 1

        output_image = (
            Path("results/saliency") / f"saliency_{class_name}_{done[class_name]}.png"
        )
        plot_saliency_map(
            original_image,
            saliency,
            title=(
                f"Saliency: {class_name} ({class_idx}) | "
                f"Gini: {gini:.3f} | Entropy: {entropy:.3f}"
            ),
            save_path=str(output_image),
        )

        results.append(
            {
                "image": image_path.name,
                "target_class_idx": int(class_idx),
                "target_class_name": class_name,
                "deletion_auc": deletion_auc,
                "insertion_auc": insertion_auc,
                "gini": gini,
                "normalized_entropy": entropy,
            }
        )

    save_saliency_metrics(results, output_path)
    return results


def run_saliency_sanity_check(
    image_path: str | Path,
    output_path: str | Path,
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

    baseline_saliency, class_idx = compute_saliency_map(model, input_batch)
    saliency_maps = [baseline_saliency]

    for _, prefixes in layer_groups[1:]:
        layer_names = [
            name
            for name, _ in model.named_modules()
            if any(name.startswith(prefix) for prefix in prefixes)
        ]

        randomize_layers_cascading(model, layer_names)

        saliency, _ = compute_saliency_map(
            model, input_batch, target_class_idx=class_idx
        )
        saliency_maps.append(saliency)

    figure, axes = plt.subplots(1, len(layer_groups) + 1, figsize=(25, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    orig_width, orig_height = original_image.size

    for axis, (title, _), saliency in zip(axes[1:], layer_groups, saliency_maps):
        saliency_pil = Image.fromarray((255 * saliency).astype(np.uint8))
        saliency_resized = saliency_pil.resize(
            (orig_width, orig_height), Image.Resampling.BILINEAR
        )
        saliency_resized_np = np.array(saliency_resized) / 255.0

        axis.imshow(saliency_resized_np, cmap="hot")
        axis.set_title(title)
        axis.axis("off")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=300)
    plt.close(figure)

    print(f"Saved Adebayo test results in: {destination}")
