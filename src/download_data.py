import os
import tarfile
import urllib.request
import shutil
from pathlib import Path

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"


def download_and_extract_imagenette(
    output_dir: str = "data/sample_images", num_samples: int = 50
):
    destination = Path(output_dir)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    archive_path = Path("imagenette2-320.tgz")

    if not archive_path.exists():
        print("Downloading Imagenette (Fast.ai)... ")
        urllib.request.urlretrieve(IMAGENETTE_URL, archive_path)
        print("Download complete!")

    print("Extracting images...")
    extracted_count = 0

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if (
                member.isfile()
                and "val/" in member.name
                and member.name.endswith(".JPEG")
            ):
                file_name = member.name.split("/")[-1]

                target_filename = destination / f"{file_name}"

                f_in = tar.extractfile(member)
                if f_in:
                    with open(target_filename, "wb") as f_out:
                        f_out.write(f_in.read())
                    extracted_count += 1
                print(f"Done: {extracted_count}")
                if extracted_count >= num_samples:
                    break

    print(
        f"Done! Prepared {extracted_count} reference images in the {output_dir} directory."
    )

    if archive_path.exists():
        archive_path.unlink()


if __name__ == "__main__":
    download_and_extract_imagenette()
