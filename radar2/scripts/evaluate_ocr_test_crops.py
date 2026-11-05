
import csv
import json
import pathlib
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data/ocr_crops/test/manifest.jsonl"
CHECKPOINT = ROOT / "models/vqa/best.pt"
OUTPUT_DIR = ROOT / "reports/ocr_clean_test"
OUTPUT_CSV = OUTPUT_DIR / "results.csv"

# YOLO checkpoint compatibility is not needed here.
sys.path.insert(0, str(ROOT))

from src.models.radar_vqa_model import RadarVQAModel
from src.inference.ocr_inference import predict_callsign


def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost,
            ))

        previous = current

    return previous[-1]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Loading OCR model...")

    model = RadarVQAModel(pretrained=False)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    records = []

    with open(MANIFEST, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    print("Test crops:", len(records))

    rows = []
    exact_matches = 0
    total_characters = 0
    total_character_errors = 0
    predictions = Counter()

    for index, record in enumerate(records, start=1):
        crop_path = Path(record["crop_path"])

        if not crop_path.is_absolute():
            crop_path = ROOT / crop_path

        actual = record["callsign"].upper()

        image = Image.open(crop_path).convert("RGB")
        result = predict_callsign(model, image)

        predicted = result["callsign"].upper()
        errors = edit_distance(actual, predicted)

        exact = predicted == actual
        exact_matches += int(exact)

        total_characters += len(actual)
        total_character_errors += errors
        predictions[predicted] += 1

        rows.append({
            "image_id": record["image_id"],
            "actual_callsign": actual,
            "predicted_callsign": predicted,
            "exact_match": exact,
            "character_errors": errors,
            "ocr_confidence": result["confidence"],
        })

        if index % 200 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)

    print("\n" + "=" * 60)
    print("CLEAN OCR TEST RESULTS")
    print("=" * 60)
    print("Test crops:", total)
    print("Exact matches:", exact_matches)
    print("Exact-match accuracy:", f"{100 * exact_matches / total:.2f}%")
    print(
        "Character accuracy:",
        f"{100 * (1 - total_character_errors / total_characters):.2f}%"
    )

    print("\nMost frequent predictions:")
    for prediction, count in predictions.most_common(15):
        print(f"{prediction or '<empty>'}: {count}")

    print("\nDetailed results:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
