
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "synthetic_complete"
ANNOTATION_DIR = DATA_DIR / "annotations"

IMAGE_SIZE = 1280

updated = 0
skipped = []

for json_path in sorted(ANNOTATION_DIR.glob("radar_*.json")):
    with json_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    image_name = Path(metadata["image"]).stem
    txt_path = ANNOTATION_DIR / f"{image_name}.txt"

    lines = []

    for contact in metadata["contacts"]:
        aircraft_bbox = contact["aircraft_bbox_xyxy"]
        callsign_bbox = contact["text_boxes"][0]["bbox_xyxy"]

        for class_id, bbox in [
            (0, aircraft_bbox),
            (1, callsign_bbox)
        ]:
            x1, y1, x2, y2 = bbox

            xc = ((x1 + x2) / 2) / IMAGE_SIZE
            yc = ((y1 + y2) / 2) / IMAGE_SIZE
            width = (x2 - x1) / IMAGE_SIZE
            height = (y2 - y1) / IMAGE_SIZE

            if not (0 <= xc <= 1 and 0 <= yc <= 1):
                raise ValueError(f"Invalid center in {image_name}")

            if not (0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"Invalid dimensions in {image_name}")

            lines.append(
                f"{class_id} {xc:.6f} {yc:.6f} "
                f"{width:.6f} {height:.6f}"
            )

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    updated += 1

print(f"Updated annotation files: {updated}")
print(f"Skipped files: {len(skipped)}")
print("Annotation correction completed.")
