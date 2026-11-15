import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

IMAGENETTE_SYNSET_TO_IMAGENET: dict[str, tuple[int, str]] = {
    "n01440764": (0, "tench"),
    "n02102040": (217, "English_springer"),
    "n02979186": (482, "cassette_player"),
    "n03000684": (491, "chain_saw"),
    "n03028079": (497, "church"),
    "n03394916": (566, "French_horn"),
    "n03417042": (569, "garbage_truck"),
    "n03425413": (571, "gas_pump"),
    "n03445777": (574, "golf_ball"),
    "n03888257": (701, "parachute"),
}


def download_and_extract_imagenette(
    output_dir: str = "data/sample_images", num_samples: int = 500
) -> dict[str, dict]:
    destination = Path(output_dir)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    archive_path = Path("imagenette2-320.tgz")

    if not archive_path.exists():
        print("Downloading Imagenette (Fast.ai)... ")
        urllib.request.urlretrieve(IMAGENETTE_URL, archive_path)
        print("Download complete!")

    print("Extracting stratified sample across Imagenette classes...")
    synsets = list(IMAGENETTE_SYNSET_TO_IMAGENET.keys())
    num_classes = len(synsets)
    base_per_class = num_samples // num_classes
    remainder = num_samples % num_classes
    quotas = {
        synset: base_per_class + (1 if idx < remainder else 0)
        for idx, synset in enumerate(synsets)
    }
    counts_per_class: dict[str, int] = {synset: 0 for synset in synsets}
    metadata: dict[str, dict] = {}
    extracted_count = 0

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if (
                not member.isfile()
                or "/val/" not in member.name
                or not member.name.endswith(".JPEG")
            ):
                continue

            parts = member.name.split("/")
            synset = parts[-2] if len(parts) >= 2 else ""
            if synset not in quotas:
                continue
            if counts_per_class[synset] >= quotas[synset]:
                continue

            file_name = parts[-1]
            target_filename = destination / file_name

            f_in = tar.extractfile(member)
            if f_in:
                with open(target_filename, "wb") as f_out:
                    f_out.write(f_in.read())
                counts_per_class[synset] += 1
                extracted_count += 1
                gt_idx, gt_label = IMAGENETTE_SYNSET_TO_IMAGENET[synset]
                metadata[file_name] = {
                    "synset": synset,
                    "ground_truth_idx": gt_idx,
                    "ground_truth_label": gt_label,
                }
                print(f"Done: {extracted_count}/{num_samples} ({gt_label})")

            if extracted_count >= num_samples:
                break

    metadata_path = destination / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Done! Prepared {extracted_count} reference images in the {output_dir} directory."
    )

    if archive_path.exists():
        archive_path.unlink()

    return metadata


if __name__ == "__main__":
    download_and_extract_imagenette()
