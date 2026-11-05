import ast
import csv
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.models.radar_vqa_model import RadarVQAModel
from src.inference.ocr_inference import predict_callsign
from src.dataset.ocr_crop_dataset import extract_callsign_crop


# Paths
CHECKPOINT_PATH = ROOT / "models/vqa/best.pt"
CSV_PATH = ROOT / "reports/yolo_ocr_100/results.csv"
IMAGE_DIR = ROOT / "data/synthetic_complete/images"

TARGET_IMAGE = "radar_00604"
TARGET_CALLSIGN = "WJA708"

DEVICE = torch.device("cpu")


def load_model():
    model = RadarVQAModel(pretrained=False)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE).eval()
    return model


def main():
    print("Loading OCR model...")

    model = load_model()

    with open(CSV_PATH, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    target = next(
        (
            row for row in rows
            if row["image_id"] == TARGET_IMAGE
            and row["actual_callsign"] == TARGET_CALLSIGN
        ),
        None,
    )

    if target is None:
        raise ValueError(
            f"Could not find {TARGET_CALLSIGN} in {TARGET_IMAGE}"
        )

    image_path = IMAGE_DIR / f"{TARGET_IMAGE}.png"
    image = Image.open(image_path).convert("RGB")

    gt_box = ast.literal_eval(target["gt_bbox"])
    pred_box = ast.literal_eval(target["predicted_bbox"])

    # Extract both crops using identical preprocessing.
    gt_crop = extract_callsign_crop(image, gt_box, padding=5)
    yolo_crop = extract_callsign_crop(image, pred_box, padding=5)

    print("\nGround truth:", TARGET_CALLSIGN)
    print("Original YOLO evaluation prediction:", target["predicted_callsign"])

    print("\nRunning OCR on ground-truth crop...")
    gt_result = predict_callsign(model, gt_crop, max_length=6)

    print("\nRunning OCR on YOLO crop...")
    yolo_result = predict_callsign(model, yolo_crop, max_length=6)

    print("\n" + "=" * 55)
    print("PAIRED OCR TEST")
    print("=" * 55)

    print("\nGround-truth crop:")
    print("  Expected:", TARGET_CALLSIGN)
    print("  Predicted:", gt_result["callsign"])
    print("  Confidence:", f'{gt_result["confidence"] * 100:.2f}%')
    print("  Character confidences:", gt_result["character_confidences"])

    print("\nYOLO crop:")
    print("  Expected:", TARGET_CALLSIGN)
    print("  Predicted:", yolo_result["callsign"])
    print("  Confidence:", f'{yolo_result["confidence"] * 100:.2f}%')
    print("  Character confidences:", yolo_result["character_confidences"])

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()