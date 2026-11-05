import csv
import json
import pathlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parents[1]

TEST_MANIFEST = ROOT / "data/splits/test.jsonl"
IMAGE_DIR = ROOT / "data/synthetic_complete/images"
ANNOTATION_DIR = ROOT / "data/synthetic_complete/annotations"

YOLO_REPO = ROOT / "models/yolov5"
YOLO_WEIGHTS = YOLO_REPO / "weights/best.pt"
VQA_CHECKPOINT = ROOT / "models/vqa/best.pt"

OUTPUT_DIR = ROOT / "reports/paddleocr_benchmark"
CSV_PATH = OUTPUT_DIR / "comparison.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

NUM_IMAGES = 100
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640
MAX_OCR_LENGTH = 6


# Windows compatibility for the Linux-saved YOLO checkpoint.
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


def calculate_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def greedy_match(gt_boxes, pred_boxes):
    candidates = []

    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = calculate_iou(gt, pred)
            if iou >= IOU_THRESHOLD:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True)

    matches = {}
    used_gt = set()
    used_pred = set()

    for iou, gi, pi in candidates:
        if gi not in used_gt and pi not in used_pred:
            matches[gi] = pi
            used_gt.add(gi)
            used_pred.add(pi)

    return matches


def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]

        for j, cb in enumerate(b, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))

        previous = current

    return previous[-1]


def normalize_callsign(text):
    # Callsigns are uppercase letters and digits.
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def read_image_ids():
    image_ids = []

    with open(TEST_MANIFEST, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            image_id = json.loads(line)["image_id"]

            if image_id not in image_ids:
                image_ids.append(image_id)

            if len(image_ids) >= NUM_IMAGES:
                break

    if not image_ids:
        raise RuntimeError(f"No test images found in {TEST_MANIFEST}")

    return image_ids


def load_models():
    print("Loading YOLO...")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    yolo = torch.hub.load(
        str(YOLO_REPO),
        "custom",
        path=str(YOLO_WEIGHTS),
        source="local",
        device=str(device),
    )
    yolo.conf = CONFIDENCE_THRESHOLD
    yolo.iou = 0.45

    print("Loading existing VQA/OCR checkpoint...")

    vqa = RadarVQAModel(pretrained=False)
    checkpoint = torch.load(
        VQA_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    vqa.load_state_dict(state_dict, strict=True)
    vqa.to(device).eval()

    print("Loading PaddleOCR...")

    paddle_ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )

    return yolo, vqa, paddle_ocr, device


def read_paddle_result(crop, paddle_ocr):
    # Both OCR systems receive the same PIL crop.
    image_array = np.asarray(crop.convert("RGB"))
    results = paddle_ocr.predict(image_array)

    texts = []
    scores = []

    for result in results:
        data = result.json.get("res", {})
        texts.extend(data.get("rec_texts", []))
        scores.extend(data.get("rec_scores", []))

    if not texts:
        return "", 0.0

    # For a callsign crop, choose the recognized text with
    # the highest recognition confidence.
    best_index = max(
        range(len(texts)),
        key=lambda i: float(scores[i]) if i < len(scores) else 0.0,
    )

    text = normalize_callsign(texts[best_index])
    score = float(scores[best_index]) if best_index < len(scores) else 0.0

    return text, score


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_ids = read_image_ids()
    yolo, vqa, paddle_ocr, device = load_models()

    print(f"\nImages: {len(image_ids)}")
    print(f"Device: {device}")
    print("Evaluating identical YOLO-detected crops.\n")

    rows = []
    start_all = time.perf_counter()

    for image_number, image_id in enumerate(image_ids, 1):
        image_path = IMAGE_DIR / f"{image_id}.png"
        annotation_path = ANNOTATION_DIR / f"{image_id}.json"

        with open(annotation_path, encoding="utf-8") as f:
            annotation = json.load(f)

        image = Image.open(image_path).convert("RGB")

        with torch.inference_mode():
            yolo_result = yolo(str(image_path), size=IMAGE_SIZE)

        detections = yolo_result.xyxy[0].detach().cpu().tolist()

        text_predictions = [
            {
                "bbox": det[:4],
                "confidence": float(det[4]),
            }
            for det in detections
            if int(det[5]) == 1
        ]

        contacts = annotation["contacts"]

        gt_boxes = [
            next(
                item["bbox_xyxy"]
                for item in contact["text_boxes"]
                if item["text"] == contact["callsign"]
            )
            for contact in contacts
        ]

        matches = greedy_match(
            gt_boxes,
            [item["bbox"] for item in text_predictions],
        )

        for gt_index, contact in enumerate(contacts):
            actual = contact["callsign"]

            row = {
                "image_id": image_id,
                "actual_callsign": actual,
                "detection_matched": gt_index in matches,
                "custom_prediction": "",
                "custom_confidence": None,
                "custom_seconds": None,
                "paddle_prediction": "",
                "paddle_confidence": None,
                "paddle_seconds": None,
            }

            # A missed detection counts as a failure for both
            # systems in the end-to-end metrics.
            if gt_index in matches:
                detection = text_predictions[matches[gt_index]]

                crop = extract_callsign_crop(
                    image,
                    detection["bbox"],
                    padding=5,
                )

                t0 = time.perf_counter()
                custom_result = predict_callsign(
                    vqa,
                    crop,
                    max_length=MAX_OCR_LENGTH,
                )
                custom_seconds = time.perf_counter() - t0

                t0 = time.perf_counter()
                paddle_text, paddle_confidence = read_paddle_result(
                    crop,
                    paddle_ocr,
                )
                paddle_seconds = time.perf_counter() - t0

                row.update({
                    "custom_prediction": normalize_callsign(
                        custom_result["callsign"]
                    ),
                    "custom_confidence": custom_result["confidence"],
                    "custom_seconds": custom_seconds,
                    "paddle_prediction": paddle_text,
                    "paddle_confidence": paddle_confidence,
                    "paddle_seconds": paddle_seconds,
                })

            row["custom_exact"] = (
                row["custom_prediction"] == actual
            )
            row["paddle_exact"] = (
                row["paddle_prediction"] == actual
            )

            row["custom_char_errors"] = edit_distance(
                actual, row["custom_prediction"]
            )
            row["paddle_char_errors"] = edit_distance(
                actual, row["paddle_prediction"]
            )

            rows.append(row)

        print(
            f"Matched detections: "
            f"{sum(r['detection_matched'] for r in rows if r['image_id'] == image_id)}"
        )

    total_seconds = time.perf_counter() - start_all

    fieldnames = list(rows[0].keys())

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    matched_rows = [r for r in rows if r["detection_matched"]]

    def summarize(prefix, evaluation_rows):
        n = len(evaluation_rows)
        exact = sum(r[f"{prefix}_exact"] for r in evaluation_rows)
        char_errors = sum(
            r[f"{prefix}_char_errors"] for r in evaluation_rows
        )
        chars = sum(len(r["actual_callsign"]) for r in evaluation_rows)

        times = [
            r[f"{prefix}_seconds"]
            for r in evaluation_rows
            if r[f"{prefix}_seconds"] is not None
        ]

        return {
            "callsigns": n,
            "exact_matches": exact,
            "exact_accuracy": exact / n if n else 0.0,
            "character_errors": char_errors,
            "character_accuracy": (
                max(0.0, 1 - char_errors / chars) if chars else 0.0
            ),
            "mean_seconds_per_crop": (
                sum(times) / len(times) if times else None
            ),
        }

    summary = {
        "images_evaluated": len(image_ids),
        "ground_truth_callsigns": total,
        "matched_yolo_callsign_detections": len(matched_rows),
        "detection_recall": len(matched_rows) / total if total else 0.0,
        "end_to_end_custom": summarize("custom", rows),
        "end_to_end_paddle": summarize("paddle", rows),
        "matched_crop_custom": summarize("custom", matched_rows),
        "matched_crop_paddle": summarize("paddle", matched_rows),
        "elapsed_seconds": total_seconds,
        "results_csv": str(CSV_PATH),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("OCR BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Images evaluated: {len(image_ids)}")
    print(f"Ground-truth callsigns: {total}")
    print(f"Matched detections: {len(matched_rows)}/{total}")
    print(f"Detection recall: {summary['detection_recall'] * 100:.2f}%")

    for label, key in [
        ("CUSTOM OCR — END TO END", "end_to_end_custom"),
        ("PADDLEOCR — END TO END", "end_to_end_paddle"),
        ("CUSTOM OCR — MATCHED CROPS", "matched_crop_custom"),
        ("PADDLEOCR — MATCHED CROPS", "matched_crop_paddle"),
    ]:
        metrics = summary[key]
        print(f"\n{label}")
        print(f"  Exact accuracy: {metrics['exact_accuracy'] * 100:.2f}%")
        print(
            f"  Character accuracy: "
            f"{metrics['character_accuracy'] * 100:.2f}%"
        )
        print(
            f"  Mean OCR time/crop: "
            f"{metrics['mean_seconds_per_crop']}"
        )

    print(f"\nElapsed time: {total_seconds:.1f}s")
    print("Detailed CSV:", CSV_PATH)
    print("Summary JSON:", SUMMARY_PATH)


if __name__ == "__main__":
    main()