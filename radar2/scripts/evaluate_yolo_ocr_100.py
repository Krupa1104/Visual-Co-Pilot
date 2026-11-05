
import csv
import json
import pathlib
import sys
from pathlib import Path

import torch
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TEST_MANIFEST = ROOT / "data/splits/test.jsonl"
IMAGE_DIR = ROOT / "data/synthetic_complete/images"
ANNOTATION_DIR = ROOT / "data/synthetic_complete/annotations"

YOLO_REPO = ROOT / "models/yolov5"
YOLO_WEIGHTS = YOLO_REPO / "weights/best.pt"
VQA_CHECKPOINT = ROOT / "models/vqa/best.pt"

OUTPUT_DIR = ROOT / "reports/yolo_ocr_100"
CSV_PATH = OUTPUT_DIR / "results.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

NUM_IMAGES = 100
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640
MAX_OCR_LENGTH = 6

# ============================================================
# WINDOWS COMPATIBILITY FOR LINUX-SAVED YOLO CHECKPOINT
# ============================================================

class CheckpointPosixPath(pathlib.PureWindowsPath):
    pass


pathlib._local = pathlib
pathlib.PosixPath = CheckpointPosixPath
sys.modules["pathlib._local"] = pathlib

sys.path.insert(0, str(YOLO_REPO))
sys.path.insert(0, str(ROOT))

from src.models.radar_vqa_model import RadarVQAModel
from src.inference.ocr_inference import predict_callsign
from src.dataset.ocr_crop_dataset import extract_callsign_crop

# ============================================================
# UTILITIES
# ============================================================

def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def greedy_match(ground_truth_boxes, predicted_boxes):
    """
    One-to-one matching using IoU.
    Returns {ground_truth_index: prediction_index}.
    """

    candidates = []

    for gt_index, gt_box in enumerate(ground_truth_boxes):
        for pred_index, pred_box in enumerate(predicted_boxes):
            score = calculate_iou(gt_box, pred_box)

            if score >= IOU_THRESHOLD:
                candidates.append((score, gt_index, pred_index))

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = {}

    for score, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue

        matches[gt_index] = pred_index
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)

    return matches


def edit_distance(a, b):
    """Levenshtein edit distance."""

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


def read_test_image_ids():
    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(
            f"Test manifest not found: {TEST_MANIFEST}"
        )

    image_ids = []

    with open(TEST_MANIFEST, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            image_id = record["image_id"]

            if image_id not in image_ids:
                image_ids.append(image_id)

            if len(image_ids) == NUM_IMAGES:
                break

    if not image_ids:
        raise RuntimeError("No image IDs found in test manifest.")

    return image_ids


# ============================================================
# LOAD MODELS ONCE
# ============================================================

def load_models(device):
    print("\nLoading YOLO checkpoint...")

    yolo = torch.hub.load(
        str(YOLO_REPO),
        "custom",
        path=str(YOLO_WEIGHTS),
        source="local",
        device=str(device),
    )

    yolo.conf = CONFIDENCE_THRESHOLD
    yolo.iou = 0.45

    print("YOLO loaded:", yolo.names)

    print("\nLoading VQA checkpoint...")

    vqa = RadarVQAModel(pretrained=False)

    checkpoint = torch.load(
        VQA_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    vqa.load_state_dict(state_dict, strict=True)

    vqa.to(device).eval()

    print("VQA checkpoint loaded successfully.")

    return yolo, vqa


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def evaluate_image(image_id, yolo, vqa):
    image_path = IMAGE_DIR / f"{image_id}.png"
    annotation_path = ANNOTATION_DIR / f"{image_id}.json"

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)

    with open(annotation_path, "r", encoding="utf-8") as file:
        annotation = json.load(file)

    image = Image.open(image_path).convert("RGB")

    # ---------------- YOLO inference ----------------

    results = yolo(str(image_path), size=IMAGE_SIZE)
    detections = results.xyxy[0].detach().cpu().tolist()

    aircraft_predictions = []
    text_predictions = []

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection

        item = {
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
        }

        if int(class_id) == 0:
            aircraft_predictions.append(item)

        elif int(class_id) == 1:
            text_predictions.append(item)

    contacts = annotation["contacts"]

    gt_aircraft_boxes = [
        contact["aircraft_bbox_xyxy"]
        for contact in contacts
    ]

    gt_callsign_boxes = []

    for contact in contacts:
        callsign_box = next(
            item["bbox_xyxy"]
            for item in contact["text_boxes"]
            if item["text"] == contact["callsign"]
        )

        gt_callsign_boxes.append(callsign_box)

    # ---------------- Detection evaluation ----------------

    aircraft_matches = greedy_match(
        gt_aircraft_boxes,
        [item["bbox"] for item in aircraft_predictions],
    )

    text_matches = greedy_match(
        gt_callsign_boxes,
        [item["bbox"] for item in text_predictions],
    )

    # ---------------- OCR inference ----------------

    ocr_results = {}

    for gt_index, pred_index in text_matches.items():
        detection = text_predictions[pred_index]

        crop = extract_callsign_crop(
            image,
            detection["bbox"],
            padding=5,
        )

        prediction = predict_callsign(
            vqa,
            crop,
            max_length=MAX_OCR_LENGTH,
        )

        ocr_results[gt_index] = prediction

    # ---------------- Per-contact evaluation ----------------

    contact_results = []

    for index, contact in enumerate(contacts):
        actual = contact["callsign"]

        prediction = ocr_results.get(index)
        predicted = (
            prediction["callsign"]
            if prediction is not None
            else ""
        )
        matched_detection = (
            text_predictions[text_matches[index]]
            if index in text_matches
            else None
        )

        contact_results.append({
            "image_id": image_id,
            "actual_callsign": actual,
            "predicted_callsign": predicted,

            "aircraft_detected": index in aircraft_matches,
            "callsign_text_detected": index in text_matches,

            "ocr_exact_match": (
                predicted == actual if prediction is not None else False
            ),
            "ocr_confidence": (
                prediction["confidence"]
                if prediction is not None else None
            ),
            "character_errors": edit_distance(actual, predicted),

            "box_iou": (
                calculate_iou(
                    gt_callsign_boxes[index],
                    matched_detection["bbox"],
                )
                if matched_detection is not None else None
            ),

            # Saved for offline crop comparison only
            "gt_bbox": gt_callsign_boxes[index],
            "predicted_bbox": (
                matched_detection["bbox"]
                if matched_detection is not None else None
            ),
        })
    return {
        "image_id": image_id,
        "ground_truth_aircraft": len(contacts),
        "aircraft_detected": len(aircraft_matches),
        "ground_truth_callsign_boxes": len(contacts),
        "callsign_text_detected": len(text_matches),
        "contact_results": contact_results,
    }


# ============================================================
# MAIN BATCH EVALUATION
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("YOLO + OCR 100-IMAGE EVALUATION")
    print("=" * 65)
    print("Device:", device)

    image_ids = read_test_image_ids()

    print("Images selected:", len(image_ids))
    print("Source: held-out test split")

    yolo, vqa = load_models(device)

    all_rows = []

    total_gt_aircraft = 0
    total_detected_aircraft = 0
    total_gt_text = 0
    total_detected_text = 0

    total_callsigns = 0
    exact_matches = 0
    total_characters = 0
    total_character_errors = 0

    for position, image_id in enumerate(image_ids, start=1):
        result = evaluate_image(image_id, yolo, vqa)

        total_gt_aircraft += result["ground_truth_aircraft"]
        total_detected_aircraft += result["aircraft_detected"]

        total_gt_text += result["ground_truth_callsign_boxes"]
        total_detected_text += result["callsign_text_detected"]

        for row in result["contact_results"]:
            all_rows.append(row)

            actual = row["actual_callsign"]
            predicted = row["predicted_callsign"]

            total_callsigns += 1
            exact_matches += int(row["ocr_exact_match"])

            total_characters += len(actual)
            total_character_errors += edit_distance(actual, predicted)

        print(
            f"[{position:03d}/{len(image_ids):03d}] "
            f"{image_id} | "
            f"Aircraft {result['aircraft_detected']}/"
            f"{result['ground_truth_aircraft']} | "
            f"Text {result['callsign_text_detected']}/"
            f"{result['ground_truth_callsign_boxes']}"
        )

    # ---------------- CSV report ----------------

        fieldnames = [
        "image_id",
        "actual_callsign",
        "predicted_callsign",
        "aircraft_detected",
        "callsign_text_detected",
        "ocr_exact_match",
        "ocr_confidence",
        "character_errors",
        "box_iou",
        "gt_bbox",
        "predicted_bbox",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # ---------------- Summary ----------------

    summary = {
        "images_evaluated": len(image_ids),
        "ground_truth_aircraft": total_gt_aircraft,
        "aircraft_detected": total_detected_aircraft,
        "aircraft_detection_recall": (
            total_detected_aircraft / total_gt_aircraft
            if total_gt_aircraft else 0
        ),
        "ground_truth_callsign_text_boxes": total_gt_text,
        "callsign_text_detected": total_detected_text,
        "callsign_text_detection_recall": (
            total_detected_text / total_gt_text
            if total_gt_text else 0
        ),
        "ground_truth_callsigns": total_callsigns,
        "exact_callsign_matches": exact_matches,
        "callsign_exact_match_accuracy": (
            exact_matches / total_callsigns
            if total_callsigns else 0
        ),
        "character_accuracy": (
            1 - total_character_errors / total_characters
            if total_characters else 0
        ),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    for key, value in summary.items():
        if "recall" in key or "accuracy" in key:
            print(f"{key}: {value * 100:.2f}%")
        else:
            print(f"{key}: {value}")

    print("\nDetailed CSV:", CSV_PATH)
    print("Summary JSON:", SUMMARY_PATH)
    print("=" * 65)


if __name__ == "__main__":
    main()
