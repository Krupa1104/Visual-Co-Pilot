
"""Generate ground-truth callsign crops for OCR training."""
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.ocr_crop_dataset import get_callsign_crops

SPLIT_DIR = ROOT / "data" / "splits"
OUTPUT_DIR = ROOT / "data" / "ocr_crops"

SPLITS = ("train", "val", "test")


def load_callsign_image_ids(split: str) -> list[str]:
    manifest_path = SPLIT_DIR / f"{split}_frames.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        frame_ids = data
    elif isinstance(data, dict) and "image_ids" in data:
        frame_ids = data["image_ids"]
    elif isinstance(data, dict) and "frames" in data:
        frame_ids = data["frames"]
    else:
        raise ValueError(
            f"Unrecognized frame manifest structure: {manifest_path}"
        )

    return [
        str(frame_id).removesuffix(".png")
        for frame_id in frame_ids
    ]


def generate_split(split: str) -> int:
    output_dir = OUTPUT_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ids = load_callsign_image_ids(split)

    total_crops = 0
    manifest_records = []

    for index, image_id in enumerate(image_ids, start=1):
        crops = get_callsign_crops(image_id)

        for crop_index, item in enumerate(crops, start=1):
            filename = f"{image_id}_{crop_index:02d}.png"
            output_path = output_dir / filename

            item["crop"].save(output_path)

            manifest_records.append({
                "image_id": image_id,
                "callsign": item["callsign"],
                "crop_path": str(output_path.relative_to(ROOT)),
            })

            total_crops += 1

        if index % 500 == 0:
            print(
                f"{split}: processed {index}/{len(image_ids)} "
                f"images | crops: {total_crops}"
            )

    manifest_path = output_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as file:
        for record in manifest_records:
            file.write(json.dumps(record) + "\n")

    print(
        f"{split.upper()} complete: "
        f"{total_crops} crops from {len(image_ids)} images"
    )

    return total_crops


def main():
    print("=" * 60)
    print("GENERATING OCR CROP DATASET")
    print("=" * 60)

    counts = {
        split: generate_split(split)
        for split in SPLITS
    }

    print("\nSummary")
    for split, count in counts.items():
        print(f"{split}: {count} crops")

    print(f"Total: {sum(counts.values())} crops")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
