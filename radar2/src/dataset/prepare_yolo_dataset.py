
"""Prepare a YOLOv5 view of the existing radar dataset.

Images are hard-linked, not copied.
Class 0: aircraft bounding box.
Class 1: callsign text bounding box (not the entire information label).

Original images, annotations and split files are never modified.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE_IMAGES = ROOT / "data" / "synthetic_complete" / "images"
SOURCE_ANNOTATIONS = ROOT / "data" / "synthetic_complete" / "annotations"
SPLIT_DIR = ROOT / "data" / "splits"

OUTPUT_DIR = ROOT / "data" / "yolo_dataset"
CONFIG_PATH = ROOT / "configs" / "radar_yolov5.yaml"

SPLITS = ("train", "val", "test")

AIRCRAFT_CLASS_ID = 0
CALLSIGN_CLASS_ID = 1


def load_splits():
    """Load existing frame IDs and check for split leakage."""
    splits = {}

    for split in SPLITS:
        path = SPLIT_DIR / f"{split}_frames.json"

        if not path.is_file():
            raise FileNotFoundError(path)

        splits[split] = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(splits[split], list):
            raise ValueError(f"{path} must contain a JSON list.")

        if len(splits[split]) != len(set(splits[split])):
            raise ValueError(f"Duplicate image IDs in {split} split.")

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Image leakage detected between splits.")

    if len(train_ids | val_ids | test_ids) != 4500:
        raise ValueError("Expected exactly 4500 unique image IDs.")

    return splits


def convert_bbox(bbox, image_width, image_height):
    """Convert pixel XYXY coordinates to normalized YOLO XYWH."""
    x1, y1, x2, y2 = map(float, bbox)

    if not (0 <= x1 < x2 <= image_width):
        raise ValueError(f"Invalid horizontal bbox: {bbox}")

    if not (0 <= y1 < y2 <= image_height):
        raise ValueError(f"Invalid vertical bbox: {bbox}")

    center_x = ((x1 + x2) / 2) / image_width
    center_y = ((y1 + y2) / 2) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    return center_x, center_y, width, height


def get_callsign_bbox(contact, image_id):
    """Find the exact text box corresponding to this contact's callsign."""
    callsign = contact["callsign"]

    matches = [
        box
        for box in contact.get("text_boxes", [])
        if box.get("text") == callsign
    ]

    if len(matches) != 1:
        raise ValueError(
            f"{image_id}: expected exactly one text box for "
            f"callsign {callsign!r}, found {len(matches)}."
        )

    return matches[0]["bbox_xyxy"]


def create_image_hardlink(source, destination):
    """Create a second directory entry referencing the same image data."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise FileExistsError(
            f"Destination exists but is not linked to the source: {destination}"
        )

    try:
        os.link(source, destination)
    except OSError as exc:
        raise RuntimeError(
            "Could not create an image hard link. "
            "The source and destination must be on the same NTFS volume. "
            "No image-copy fallback was used."
        ) from exc


def prepare_split(split, image_ids):
    image_count = 0
    aircraft_count = 0
    callsign_count = 0
    total_boxes = 0
    empty_count = 0

    image_output = OUTPUT_DIR / "images" / split
    label_output = OUTPUT_DIR / "labels" / split

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        image_path = SOURCE_IMAGES / f"{image_id}.png"
        annotation_path = SOURCE_ANNOTATIONS / f"{image_id}.json"

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)

        annotation = json.loads(
            annotation_path.read_text(encoding="utf-8")
        )

        image_width = annotation["width"]
        image_height = annotation["height"]

        if image_width != 1280 or image_height != 1280:
            raise ValueError(f"Unexpected dimensions for {image_id}.")

        contacts = annotation.get("contacts", [])

        if annotation["contact_count"] != len(contacts):
            raise ValueError(f"Contact count mismatch in {image_id}.")

        lines = []

        for contact in contacts:
            aircraft_box = convert_bbox(
                contact["aircraft_bbox_xyxy"],
                image_width,
                image_height,
            )

            callsign_bbox = convert_bbox(
                get_callsign_bbox(contact, image_id),
                image_width,
                image_height,
            )

            lines.append(
                f"{AIRCRAFT_CLASS_ID} "
                + " ".join(f"{value:.8f}" for value in aircraft_box)
            )

            lines.append(
                f"{CALLSIGN_CLASS_ID} "
                + " ".join(f"{value:.8f}" for value in callsign_bbox)
            )

        create_image_hardlink(
            image_path,
            image_output / image_path.name,
        )

        # Rewrite only the YOLO label file.
        # Empty files are valid for images with zero aircraft.
        label_path = label_output / f"{image_id}.txt"
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

        image_count += 1
        aircraft_count += len(contacts)
        callsign_count += len(contacts)
        total_boxes += len(lines)

        if not contacts:
            empty_count += 1

    print(
        f"{split.upper():5} | Images: {image_count:4} | "
        f"Aircraft: {aircraft_count:5} | "
        f"Callsigns: {callsign_count:5} | "
        f"Total boxes: {total_boxes:5} | "
        f"Empty images: {empty_count:4}"
    )

    return (
        image_count,
        aircraft_count,
        callsign_count,
        total_boxes,
        empty_count,
    )


def write_yaml():
    """Create the YOLOv5 dataset configuration."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = OUTPUT_DIR.as_posix()

    config = (
        f"path: '{dataset_path}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 2\n"
        "names:\n"
        "  0: aircraft\n"
        "  1: callsign_text\n"
    )

    CONFIG_PATH.write_text(config, encoding="utf-8")


def main():
    print("=" * 70)
    print("YOLOv5 DATASET PREPARATION")
    print("Aircraft + exact callsign text detection")
    print("Reusing existing images through hard links")
    print("=" * 70)

    splits = load_splits()

    totals = [0, 0, 0, 0, 0]

    for split in SPLITS:
        result = prepare_split(split, splits[split])
        totals = [a + b for a, b in zip(totals, result)]

    if totals[0] != 4500:
        raise RuntimeError("Expected 4500 prepared images.")

    if totals[1] != 18918 or totals[2] != 18918 or totals[3] != 37836:
        raise RuntimeError(
            "Unexpected bounding-box totals. "
            f"Aircraft: {totals[1]}, callsigns: {totals[2]}, "
            f"total boxes: {totals[3]}"
        )

    write_yaml()

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETED")
    print("=" * 70)
    print("Images referenced:", totals[0])
    print("Aircraft boxes:", totals[1])
    print("Callsign text boxes:", totals[2])
    print("Total YOLO boxes:", totals[3])
    print("Empty images:", totals[4])
    print("Dataset directory:", OUTPUT_DIR)
    print("YOLO configuration:", CONFIG_PATH)
    print("Original dataset: untouched")
    print("Image bytes copied: no")


if __name__ == "__main__":
    main()
