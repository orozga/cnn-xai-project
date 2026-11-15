import argparse

from src.evaluate import run_comparative_pipeline, run_sanity_check
from src.model import XAIResNetModel
from src.utils import get_imagenet_mapping, get_sample_images_paths
from src.xai_methods import AVAILABLE_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified XAI evaluation pipeline for ResNet-50 (Phase I & Phase II)."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(AVAILABLE_METHODS),
        choices=list(AVAILABLE_METHODS),
        help="XAI methods to evaluate (default: all 4 core methods).",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/sample_images",
        help="Directory containing evaluation images.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of Deletion/Insertion curve steps.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on the number of images to process.",
    )
    parser.add_argument(
        "--scorecam-max-channels",
        type=int,
        default=None,
        help="Optional cap on top activated channels for Score-CAM.",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip cascading parameter randomization sanity checks.",
    )
    parser.add_argument(
        "--sanity-check-limit",
        type=int,
        default=5,
        help="Number of sample images per method to run sanity checks on (0 = all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = get_sample_images_paths(args.image_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {args.image_dir}. Run `python -m src.download_data` first."
        )

    if args.max_images is not None and args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    print(
        f"Found {len(image_paths)} images. "
        f"Running unified XAI pipeline for methods: {', '.join(args.methods)}"
    )
    model_wrapper = XAIResNetModel()
    summaries = run_comparative_pipeline(
        model_wrapper,
        methods=args.methods,
        image_dir=args.image_dir,
        steps=args.steps,
        max_images=args.max_images,
        scorecam_max_channels=args.scorecam_max_channels,
    )

    for method_key, summary in summaries.items():
        print(
            f"[{method_key}] Completed analysis for {summary['image_count']} images -> "
            f"results/{method_key}/metrics_summary.json"
        )
    print(
        "Comparative report saved to results/comparison/RAPORT_FAZA_2.md "
        "and results/comparison/comparison_summary.json."
    )

    if not args.skip_sanity_check:
        sanity_paths = (
            image_paths[: args.sanity_check_limit]
            if args.sanity_check_limit > 0
            else image_paths
        )
        class_mapping = get_imagenet_mapping()
        print(
            f"Running Adebayo cascading randomization sanity checks on {len(sanity_paths)} images..."
        )
        for method_key in args.methods:
            sanity_counts: dict[str, int] = {}
            for image_path in sanity_paths:
                logits = model_wrapper.predict(
                    __import__("src.utils", fromlist=["load_and_preprocess_image"])
                    .load_and_preprocess_image(image_path)[1]
                    .to(model_wrapper.device)
                )
                class_idx = int(logits.argmax(dim=1).item())
                class_name = class_mapping.get(class_idx, "Unknown")
                sanity_counts[class_name] = sanity_counts.get(class_name, 0) + 1
                sanity_output = (
                    f"results/{method_key}/sanity_check_{class_name}_"
                    f"{sanity_counts[class_name]}.png"
                )
                run_sanity_check(
                    image_path,
                    sanity_output,
                    method=method_key,
                    scorecam_max_channels=args.scorecam_max_channels or 128,
                )

    model_wrapper.remove_hooks()
    print("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
