
"""
Ground-truth callsign crop extraction for OCR training.

IMPORTANT:
Annotations are used only to prepare training crops.
Never use this module during model inference.
"""

import json
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "synthetic_complete" / "images"
ANNOTATION_DIR = ROOT / "data" / "synthetic_complete" / "annotations"

CROP_WIDTH = 224
CROP_HEIGHT = 96
PADDING = 5


def extract_callsign_crop(
    image: Image.Image,
    bbox: list[int],
    padding: int = PADDING,
) -> Image.Image:
    """Extract and resize a padded callsign crop without distorting it."""

    if len(bbox) != 4:
        raise ValueError(f"Expected four bbox coordinates, got {bbox}")

    x1, y1, x2, y2 = map(int, bbox)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box: {bbox}")

    image_width, image_height = image.size

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)

    crop = image.crop((x1, y1, x2, y2)).convert("RGB")

    # Preserve aspect ratio and fit the crop into the target dimensions.
    crop = ImageOps.contain(
        crop,
        (CROP_WIDTH, CROP_HEIGHT),
        method=Image.Resampling.LANCZOS,
    )

    # Center the crop on a black canvas.
    canvas = Image.new("RGB", (CROP_WIDTH, CROP_HEIGHT), (0, 0, 0))
    offset = (
        (CROP_WIDTH - crop.width) // 2,
        (CROP_HEIGHT - crop.height) // 2,
    )
    canvas.paste(crop, offset)

    return canvas


def get_callsign_crops(image_id: str):
    """
    Return all annotated callsign crops for one image.

    This function is intended for training-data preparation only.
    """

    image_path = IMAGE_DIR / f"{image_id}.png"
    annotation_path = ANNOTATION_DIR / f"{image_id}.json"

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    annotation = json.loads(
        annotation_path.read_text(encoding="utf-8")
    )

    results = []

    for contact in annotation.get("contacts", []):
        callsign = contact["callsign"]

        matching_boxes = [
            box
            for box in contact.get("text_boxes", [])
            if box.get("text") == callsign
        ]

        if len(matching_boxes) != 1:
            raise ValueError(
                f"{image_id}: expected exactly one callsign box "
                f"for {callsign}, found {len(matching_boxes)}"
            )

        bbox = matching_boxes[0]["bbox_xyxy"]

        crop = extract_callsign_crop(image, bbox)

        results.append({
            "image_id": image_id,
            "callsign": callsign,
            "crop": crop,
        })

    return results


if __name__ == "__main__":
    sample = get_callsign_crops("radar_00001")

    print("Image:", "radar_00001")
    print("Extracted crops:", len(sample))

    for item in sample:
        print("Callsign:", item["callsign"])
        print("Crop size:", item["crop"].size)

    if sample:
        preview_path = ROOT / "outputs" / "ocr_crop_test.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        sample[0]["crop"].save(preview_path)
        print("Saved:", preview_path)