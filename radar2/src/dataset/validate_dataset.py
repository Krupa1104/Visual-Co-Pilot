
import json
from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "synthetic" / "images"
ANNO_DIR = ROOT / "data" / "synthetic" / "annotations"
MANIFEST = ROOT / "data" / "synthetic" / "all_frames.json"

EXPECTED_IMAGES = 4000
EXPECTED_SIZE = (1280, 1280)


def main():
    print("=" * 60)
    print("SYNTHETIC RADAR DATASET VALIDATION")
    print("=" * 60)

    images = sorted(IMAGE_DIR.glob("radar_*.png"))
    txt_files = sorted(ANNO_DIR.glob("radar_*.txt"))
    json_files = sorted(ANNO_DIR.glob("radar_*.json"))

    print(f"\nImages found       : {len(images)}")
    print(f"YOLO labels found  : {len(txt_files)}")
    print(f"JSON annotations  : {len(json_files)}")

    errors = []

    if len(images) != EXPECTED_IMAGES:
        errors.append(f"Expected {EXPECTED_IMAGES} images, found {len(images)}")

    if len(txt_files) != len(images):
        errors.append("Image and YOLO annotation counts differ")

    if len(json_files) != len(images):
        errors.append("Image and JSON annotation counts differ")

    # Check image readability and resolution.
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                if image.size != EXPECTED_SIZE:
                    errors.append(
                        f"{image_path.name}: incorrect size {image.size}"
                    )
        except Exception as exc:
            errors.append(f"{image_path.name}: image error: {exc}")

    # Check YOLO annotation format.
    class_counts = Counter()
    label_count = 0

    for label_path in txt_files:
        with label_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                parts = line.strip().split()

                if not parts:
                    continue

                if len(parts) != 5:
                    errors.append(
                        f"{label_path.name}:{line_number}: expected 5 values"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, width, height = map(float, parts[1:])
                except ValueError:
                    errors.append(
                        f"{label_path.name}:{line_number}: invalid numeric value"
                    )
                    continue

                if class_id < 0:
                    errors.append(f"{label_path.name}: negative class ID")

                if not all(0 <= v <= 1 for v in (x, y, width, height)):
                    errors.append(
                        f"{label_path.name}:{line_number}: "
                        "coordinates outside [0, 1]"
                    )

                if width <= 0 or height <= 0:
                    errors.append(
                        f"{label_path.name}:{line_number}: invalid box size"
                    )

                class_counts[class_id] += 1
                label_count += 1

    # Check JSON annotations.
    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as file:
                json.load(file)
        except Exception as exc:
            errors.append(f"{json_path.name}: invalid JSON: {exc}")

    # Check manifest.
    if not MANIFEST.exists():
        errors.append("Manifest file is missing")
    else:
        try:
            with MANIFEST.open("r", encoding="utf-8") as file:
                manifest = json.load(file)

            if not isinstance(manifest, list):
                errors.append("Manifest is not a JSON list")
            elif len(manifest) != EXPECTED_IMAGES:
                errors.append(
                    f"Manifest contains {len(manifest)} frames, "
                    f"expected {EXPECTED_IMAGES}"
                )
        except Exception as exc:
            errors.append(f"Manifest error: {exc}")

    print(f"\nTotal YOLO objects : {label_count}")
    print("\nYOLO class distribution:")

    for class_id, count in sorted(class_counts.items()):
        print(f"  Class {class_id}: {count}")

    print("\nValidation errors:", len(errors))

    for error in errors[:30]:
        print("  ERROR:", error)

    if len(errors) > 30:
        print(f"  ... and {len(errors) - 30} more errors")

    if errors:
        print("\nVALIDATION FAILED")
    else:
        print("\nVALIDATION PASSED")

    print("=" * 60)


if __name__ == "__main__":
    main()
