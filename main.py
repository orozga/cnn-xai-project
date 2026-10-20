from src.evaluate import run_full_evaluation, run_saliency_sanity_check
from src.model import XAIResNetModel
from src.utils import get_sample_images_paths, save_metrics_summary


def main() -> None:
    image_dir = "data/sample_images"
    image_paths = get_sample_images_paths(image_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {image_dir}. Add images before running evaluation."
        )

    print(f"Found {len(image_paths)} images. Starting full Saliency evaluation.")
    model_wrapper = XAIResNetModel()
    results = run_full_evaluation(model_wrapper, image_dir=image_dir)
    save_metrics_summary(results, "results/saliency/metrics_summary.json")
    print(
        f"Completed Saliency analysis for {len(results)} images. "
        "Metrics saved to results/saliency/metrics_results.json."
    )

    print("Starting Saliency sanity checks for all images.")
    sanity_counts: dict[str, int] = {}
    results_by_image = {result["image"]: result for result in results}
    for image_path in sorted(image_paths):
        result = results_by_image[image_path.name]
        class_name = result["target_class_name"]
        sanity_counts[class_name] = sanity_counts.get(class_name, 0) + 1
        sanity_output = (
            f"results/saliency/sanity_check_{class_name}_"
            f"{sanity_counts[class_name]}.png"
        )
        run_saliency_sanity_check(image_path, sanity_output)

    print(
        "Completed sanity checks for all images. "
        "Summary saved to results/saliency/metrics_summary.json."
    )


if __name__ == "__main__":
    main()
