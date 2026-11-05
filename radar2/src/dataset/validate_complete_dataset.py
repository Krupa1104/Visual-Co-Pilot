
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "synthetic_complete"
IMAGE_DIR = DATA_DIR / "images"
ANNO_DIR = DATA_DIR / "annotations"
MANIFEST = DATA_DIR / "all_frames.json"

EXPECTED_IMAGES = 4500
EXPECTED_SIZE = (1280, 1280)
EXPECTED_QUOTA = 500
TOLERANCE = 1.1e-6

errors = []
warnings = []


def error(message):
    errors.append(message)


def sha256_file(path):
    with Image.open(path) as image:
        return hashlib.sha256(image.tobytes()).hexdigest()


def valid_bbox(bbox, filename, description):
    if not isinstance(bbox, list) or len(bbox) != 4:
        error(f"{filename}: invalid {description} format")
        return False

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError):
        error(f"{filename}: non-numeric {description}")
        return False

    if not (0 <= x1 < x2 <= EXPECTED_SIZE[0]):
        error(f"{filename}: invalid {description} x-coordinates")
        return False

    if not (0 <= y1 < y2 <= EXPECTED_SIZE[1]):
        error(f"{filename}: invalid {description} y-coordinates")
        return False

    return True


def bbox_to_yolo(bbox):
    x1, y1, x2, y2 = map(float, bbox)

    return (
        (x1 + x2) / (2 * EXPECTED_SIZE[0]),
        (y1 + y2) / (2 * EXPECTED_SIZE[1]),
        (x2 - x1) / EXPECTED_SIZE[0],
        (y2 - y1) / EXPECTED_SIZE[1],
    )


def close_values(actual, expected):
    return all(
        abs(a - e) <= TOLERANCE
        for a, e in zip(actual, expected)
    )


def main():
    print("=" * 68)
    print("COMPREHENSIVE RADAR DATASET VALIDATION")
    print("=" * 68)

    images = sorted(IMAGE_DIR.glob("radar_*.png"))
    labels = sorted(ANNO_DIR.glob("radar_*.txt"))
    annotations = sorted(ANNO_DIR.glob("radar_*.json"))

    print(f"\nImages              : {len(images)}")
    print(f"YOLO label files    : {len(labels)}")
    print(f"JSON annotation files: {len(annotations)}")

    if len(images) != EXPECTED_IMAGES:
        error(f"Expected {EXPECTED_IMAGES} images, found {len(images)}")

    image_map = {p.stem: p for p in images}
    label_map = {p.stem: p for p in labels}
    annotation_map = {p.stem: p for p in annotations}

    if len(image_map) != len(images):
        error("Duplicate image filenames detected")

    if len(label_map) != len(labels):
        error("Duplicate YOLO filenames detected")

    if len(annotation_map) != len(annotations):
        error("Duplicate JSON filenames detected")

    if set(image_map) != set(label_map):
        error("Image and YOLO filename sets do not match")

    if set(image_map) != set(annotation_map):
        error("Image and JSON filename sets do not match")

    if not MANIFEST.exists():
        error("Manifest file is missing")
        manifest = []
    else:
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            if not isinstance(manifest, list):
                error("Manifest must be a JSON list")
                manifest = []
        except Exception as exc:
            error(f"Manifest cannot be read: {exc}")
            manifest = []

    print(f"Manifest entries    : {len(manifest)}")

    if len(manifest) != EXPECTED_IMAGES:
        error(
            f"Expected {EXPECTED_IMAGES} manifest entries, "
            f"found {len(manifest)}"
        )

    manifest_map = {}

    for entry in manifest:
        if not isinstance(entry, dict):
            error("Manifest contains a non-object entry")
            continue

        name = entry.get("image")

        if not isinstance(name, str):
            error("Manifest entry has invalid image filename")
            continue

        if name in manifest_map:
            error(f"Duplicate manifest entry: {name}")

        manifest_map[name] = entry

    if set(manifest_map) != {p.name for p in images}:
        error("Manifest image filenames do not match image files")

    # Dataset-wide counters.
    count_distribution = Counter()
    class_distribution = Counter()
    image_hashes = {}
    scene_hashes = {}
    total_contacts = 0
    total_text_boxes = 0
    zero_aircraft_images = 0

    for index, image_path in enumerate(images, start=1):
        filename = image_path.name
        stem = image_path.stem

        if index % 500 == 0:
            print(f"Checked {index}/{len(images)} images...")

        # ----------------------------------------------------
        # Image integrity and image hash.
        # ----------------------------------------------------
        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                if image.size != EXPECTED_SIZE:
                    error(f"{filename}: incorrect dimensions {image.size}")

            image_hash = sha256_file(image_path)

            if image_hash in image_hashes:
                error(
                    f"{filename}: duplicate image content with "
                    f"{image_hashes[image_hash]}"
                )
            else:
                image_hashes[image_hash] = filename

        except Exception as exc:
            error(f"{filename}: image integrity error: {exc}")
            continue

        # ----------------------------------------------------
        # Load JSON annotation.
        # ----------------------------------------------------
        annotation_path = annotation_map.get(stem)

        if annotation_path is None:
            continue

        try:
            annotation = json.loads(
                annotation_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            error(f"{annotation_path.name}: invalid JSON: {exc}")
            continue

        manifest_entry = manifest_map.get(filename)

        if manifest_entry is None:
            error(f"{filename}: missing manifest entry")
            continue

        # ----------------------------------------------------
        # Manifest and JSON consistency.
        # ----------------------------------------------------
        if annotation != manifest_entry:
            error(f"{filename}: manifest and JSON annotation differ")

        if annotation.get("image") != filename:
            error(f"{filename}: JSON image filename mismatch")

        if annotation.get("width") != EXPECTED_SIZE[0]:
            error(f"{filename}: incorrect JSON width")

        if annotation.get("height") != EXPECTED_SIZE[1]:
            error(f"{filename}: incorrect JSON height")

        contacts = annotation.get("contacts")

        if not isinstance(contacts, list):
            error(f"{filename}: contacts must be a list")
            continue

        contact_count = annotation.get("contact_count")

        if contact_count != len(contacts):
            error(f"{filename}: contact_count does not match contacts")

        if not isinstance(contact_count, int) or contact_count < 0:
            error(f"{filename}: invalid contact_count")
            continue

        count_category = min(contact_count, 8)
        count_distribution[count_category] += 1
        total_contacts += contact_count

        if contact_count == 0:
            zero_aircraft_images += 1

        # ----------------------------------------------------
        # Image hash and scene hash in manifest.
        # ----------------------------------------------------
        if annotation.get("image_sha256") != image_hash:
            error(f"{filename}: image SHA-256 does not match manifest")

        scene_hash = annotation.get("scene_sha256")

        if not isinstance(scene_hash, str) or len(scene_hash) != 64:
            error(f"{filename}: invalid scene SHA-256")
        elif scene_hash in scene_hashes:
            error(
                f"{filename}: duplicate scene hash with "
                f"{scene_hashes[scene_hash]}"
            )
        else:
            scene_hashes[scene_hash] = filename

        # ----------------------------------------------------
        # Validate contact metadata and bounding boxes.
        # ----------------------------------------------------
        expected_yolo_rows = []

        callsigns = set()

        for contact_index, contact in enumerate(contacts):
            prefix = f"{filename}, contact {contact_index}"

            callsign = contact.get("callsign")
            altitude = contact.get("altitude")
            heading = contact.get("heading")
            position = contact.get("position")

            if not isinstance(callsign, str) or not callsign:
                error(f"{prefix}: invalid callsign")
            elif callsign in callsigns:
                error(f"{prefix}: duplicate callsign in same image")
            else:
                callsigns.add(callsign)

            if not isinstance(altitude, int) or not 0 <= altitude <= 100000:
                error(f"{prefix}: invalid altitude")

            if not isinstance(heading, int) or not 0 <= heading <= 359:
                error(f"{prefix}: invalid heading")

            if (
                not isinstance(position, list)
                or len(position) != 2
                or not all(
                    isinstance(v, (int, float))
                    and 0 <= v < EXPECTED_SIZE[0]
                    for v in position
                )
            ):
                error(f"{prefix}: invalid position")

            aircraft_bbox = contact.get("aircraft_bbox_xyxy")
            label_bbox = contact.get("label_bbox_xyxy")

            aircraft_ok = valid_bbox(
                aircraft_bbox, filename, "aircraft bbox"
            )
            label_ok = valid_bbox(
                label_bbox, filename, "label bbox"
            )

            if aircraft_ok:
                expected_yolo_rows.append(
                    (0, bbox_to_yolo(aircraft_bbox))
                )

            if label_ok:
                expected_yolo_rows.append(
                    (1, bbox_to_yolo(label_bbox))
                )

            text_boxes = contact.get("text_boxes")

            if not isinstance(text_boxes, list) or len(text_boxes) != 3:
                error(f"{prefix}: expected exactly three text boxes")
                continue

            for text_box in text_boxes:
                if not isinstance(text_box, dict):
                    error(f"{prefix}: invalid text box object")
                    continue

                if not isinstance(text_box.get("text"), str):
                    error(f"{prefix}: invalid text box text")

                if valid_bbox(
                    text_box.get("bbox_xyxy"),
                    filename,
                    "text bbox",
                ):
                    total_text_boxes += 1

        # ----------------------------------------------------
        # YOLO labels: validate syntax and exact bbox mapping.
        # Class 0 = aircraft; class 1 = information label.
        # ----------------------------------------------------
        label_path = label_map.get(stem)

        if label_path is None:
            continue

        actual_yolo_rows = []

        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()

            for line_number, line in enumerate(lines, start=1):
                parts = line.split()

                if len(parts) != 5:
                    error(
                        f"{label_path.name}:{line_number}: "
                        "expected five YOLO values"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    values = tuple(map(float, parts[1:]))
                except ValueError:
                    error(
                        f"{label_path.name}:{line_number}: "
                        "invalid numeric values"
                    )
                    continue

                if class_id not in (0, 1):
                    error(
                        f"{label_path.name}:{line_number}: "
                        f"unexpected class ID {class_id}"
                    )
                    continue

                if not all(0 <= value <= 1 for value in values):
                    error(
                        f"{label_path.name}:{line_number}: "
                        "normalized values outside [0,1]"
                    )
                    continue

                if values[2] <= 0 or values[3] <= 0:
                    error(
                        f"{label_path.name}:{line_number}: "
                        "non-positive box dimensions"
                    )
                    continue

                cx, cy, width, height = values

                if (
                    cx - width / 2 < -TOLERANCE
                    or cy - height / 2 < -TOLERANCE
                    or cx + width / 2 > 1 + TOLERANCE
                    or cy + height / 2 > 1 + TOLERANCE
                ):
                    error(
                        f"{label_path.name}:{line_number}: "
                        "bounding box extends outside image"
                    )

                class_distribution[class_id] += 1
                actual_yolo_rows.append((class_id, values))

        except Exception as exc:
            error(f"{label_path.name}: cannot read YOLO file: {exc}")
            continue

        if len(actual_yolo_rows) != len(expected_yolo_rows):
            error(
                f"{filename}: YOLO object count does not match "
                "JSON bounding-box count"
            )
        else:
            for row_index, (actual, expected) in enumerate(
                zip(actual_yolo_rows, expected_yolo_rows)
            ):
                actual_class, actual_values = actual
                expected_class, expected_values = expected

                if actual_class != expected_class:
                    error(
                        f"{filename}: YOLO class ordering mismatch "
                        f"at row {row_index + 1}"
                    )
                    continue

                if not close_values(actual_values, expected_values):
                    error(
                        f"{filename}: YOLO bbox does not match "
                        f"JSON bbox at row {row_index + 1}"
                    )

    # --------------------------------------------------------
    # Exact aircraft-count quotas.
    # --------------------------------------------------------
    print("\nAircraft-count distribution:")

    for category in range(8):
        actual = count_distribution[category]
        print(f"  {category:<4}: {actual}")
        if actual != EXPECTED_QUOTA:
            error(
                f"Category {category}: expected {EXPECTED_QUOTA}, "
                f"found {actual}"
            )

    actual_8_plus = count_distribution[8]
    print(f"  {'8+':<4}: {actual_8_plus}")

    if actual_8_plus != EXPECTED_QUOTA:
        error(
            f"Category 8+: expected {EXPECTED_QUOTA}, "
            f"found {actual_8_plus}"
        )

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------
    print("\nYOLO class distribution:")
    for class_id in (0, 1):
        print(f"  Class {class_id}: {class_distribution[class_id]}")

    print(f"\nTotal aircraft contacts : {total_contacts}")
    print(f"Total text boxes        : {total_text_boxes}")
    print(f"Zero-aircraft images    : {zero_aircraft_images}")
    print(f"Unique image hashes     : {len(image_hashes)}")
    print(f"Unique scene hashes     : {len(scene_hashes)}")

    print(f"\nValidation errors: {len(errors)}")

    for item in errors[:50]:
        print("  ERROR:", item)

    if len(errors) > 50:
        print(f"  ... and {len(errors) - 50} more errors")

    if warnings:
        print(f"\nWarnings: {len(warnings)}")
        for item in warnings:
            print("  WARNING:", item)

    if errors:
        print("\nCOMPREHENSIVE VALIDATION FAILED")
    else:
        print("\nCOMPREHENSIVE VALIDATION PASSED")

    print("=" * 68)


if __name__ == "__main__":
    main()
