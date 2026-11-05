
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.radar_vqa_model import RadarVQAModel
from src.inference.ocr_inference import predict_callsign
from src.dataset.ocr_crop_dataset import extract_callsign_crop


CHECKPOINT = ROOT / "models/vqa/best.pt"
MANIFEST = ROOT / "data/ocr_crops/val/manifest.jsonl"
OUTPUT_DIR = ROOT / "reports/ocr_autoregressive"
DEVICE = torch.device("cpu")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading OCR checkpoint...")
    model = RadarVQAModel(pretrained=False)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(
        checkpoint.get("model_state_dict", checkpoint),
        strict=True,
    )
    model.to(DEVICE).eval()

    records = []
    with open(MANIFEST, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    print(f"Validation crops: {len(records)}")
    print("Running autoregressive evaluation...")

    exact_correct = 0
    total_characters = 0
    total_character_errors = 0
    position_correct = [0] * 6
    position_total = [0] * 6
    results = []

    for index, record in enumerate(records, start=1):
        crop_path = ROOT / record["crop_path"]

        with Image.open(crop_path) as image:
            crop = image.convert("RGB")

        prediction = predict_callsign(
            model,
            crop,
            max_length=6,
        )

        actual = record["callsign"]
        predicted = prediction["callsign"]

        if predicted == actual:
            exact_correct += 1

        total_characters += len(actual)

        errors = sum(
            a != b
            for a, b in zip(actual, predicted)
        ) + abs(len(actual) - len(predicted))

        total_character_errors += errors

        for position in range(6):
            if position < len(actual):
                position_total[position] += 1

                if (
                    position < len(predicted)
                    and actual[position] == predicted[position]
                ):
                    position_correct[position] += 1

        results.append({
            "image_id": record["image_id"],
            "actual": actual,
            "predicted": predicted,
            "exact_match": predicted == actual,
            "confidence": prediction["confidence"],
            "character_errors": errors,
        })

        if index % 200 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)}")

    exact_accuracy = exact_correct / max(len(records), 1) * 100
    character_accuracy = (
        1 - total_character_errors / max(total_characters, 1)
    ) * 100

    print("\n" + "=" * 55)
    print("AUTOREGRESSIVE OCR VALIDATION RESULTS")
    print("=" * 55)
    print("Total crops:", len(records))
    print("Exact matches:", exact_correct)
    print(f"Exact callsign accuracy: {exact_accuracy:.2f}%")
    print(f"Character accuracy: {character_accuracy:.2f}%")

    print("\nPer-position accuracy:")
    for position in range(6):
        accuracy = (
            position_correct[position]
            / max(position_total[position], 1)
            * 100
        )
        print(
            f"Position {position + 1}: "
            f"{position_correct[position]}/{position_total[position]} "
            f"({accuracy:.2f}%)"
        )

    with open(
        OUTPUT_DIR / "results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    summary = {
        "total_crops": len(records),
        "exact_matches": exact_correct,
        "exact_accuracy": round(exact_accuracy, 2),
        "character_accuracy": round(character_accuracy, 2),
        "position_accuracy": [
            round(
                position_correct[i] / max(position_total[i], 1) * 100,
                2,
            )
            for i in range(6)
        ],
    }

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nReports saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
