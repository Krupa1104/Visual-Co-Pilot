
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import random

# ------------------------------------------------------------
# PATHS AND SETTINGS
# ------------------------------------------------------------

RADAR_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RADAR_ROOT.parent

DEFAULT_DATASET = RADAR_ROOT / "data" / "synthetic_complete"
DEFAULT_YOLO_REPO = RADAR_ROOT / "models" / "yolov5"

CONFIDENCE_THRESHOLDS = [0.10, 0.25, 0.40, 0.55, 0.70]
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 1280

CLASS_NAMES = {
    0: "aircraft",
    1: "text",
}


# ------------------------------------------------------------
# IOU
# ------------------------------------------------------------

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )
    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# ------------------------------------------------------------
# LOAD GROUND TRUTH
# ------------------------------------------------------------

def load_ground_truth(label_path, image_width, image_height):
    ground_truth = []

    if not label_path.exists():
        return ground_truth

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))
        cx, cy, width, height = map(float, parts[1:])

        x1 = (cx - width / 2) * image_width
        y1 = (cy - height / 2) * image_height
        x2 = (cx + width / 2) * image_width
        y2 = (cy + height / 2) * image_height

        ground_truth.append({
            "class_id": class_id,
            "bbox": [x1, y1, x2, y2],
        })

    return ground_truth


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------



def load_model(repo_path, weights_path):
    if not repo_path.exists():
        raise FileNotFoundError(
            f"YOLOv5 repository not found: {repo_path}"
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}"
        )

    print(f"Loading YOLOv5 repository: {repo_path}")
    print(f"Loading checkpoint: {weights_path}")

    import pathlib
    import sys
    import types

    # Map Linux checkpoint paths to Windows paths during deserialization.
    pathlib_local = types.ModuleType("pathlib._local")
    pathlib_local.WindowsPath = pathlib.WindowsPath
    pathlib_local.PosixPath = pathlib.WindowsPath
    sys.modules["pathlib._local"] = pathlib_local

    original_posix_path = pathlib.PosixPath

    try:
        pathlib.PosixPath = pathlib.WindowsPath

        model = torch.hub.load(
            str(repo_path),
            "custom",
            path=str(weights_path),
            source="local",
            force_reload=False,
        )
    finally:
        pathlib.PosixPath = original_posix_path

    model.conf = 0.001
    model.iou = IOU_THRESHOLD
    model.max_det = 300
    model.eval()

    return model


# ------------------------------------------------------------
# RUN INFERENCE ONCE PER IMAGE
# ------------------------------------------------------------

def predict_image(model, image_path):
    results = model(str(image_path), size=IMAGE_SIZE)

    predictions = []

    for row in results.xyxy[0].detach().cpu().numpy():
        x1, y1, x2, y2, confidence, class_id = row

        predictions.append({
            "class_id": int(class_id),
            "confidence": float(confidence),
            "bbox": [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ],
        })

    return predictions


# ------------------------------------------------------------
# MATCH PREDICTIONS TO GROUND TRUTH
# ------------------------------------------------------------

def match_predictions(
    predictions,
    ground_truth,
    class_id,
    confidence_threshold,
    iou_threshold,
):
    preds = [
        p for p in predictions
        if p["class_id"] == class_id
        and p["confidence"] >= confidence_threshold
    ]

    truths = [
        g for g in ground_truth
        if g["class_id"] == class_id
    ]

    preds.sort(
        key=lambda p: p["confidence"],
        reverse=True,
    )

    matched_truths = set()
    true_positives = 0
    false_positives = 0
    duplicates = 0
    details = []

    for pred in preds:
        best_iou = 0.0
        best_index = None

        for index, truth in enumerate(truths):
            if index in matched_truths:
                continue

            iou = calculate_iou(
                pred["bbox"],
                truth["bbox"],
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            matched_truths.add(best_index)
            true_positives += 1

            details.append({
                "status": "TP",
                "confidence": pred["confidence"],
                "iou": best_iou,
                "bbox": pred["bbox"],
            })

        else:
            false_positives += 1

            # Check whether this prediction overlaps an
            # already-matched ground-truth box.
            duplicate = any(
                calculate_iou(pred["bbox"], truth["bbox"])
                >= iou_threshold
                for index, truth in enumerate(truths)
                if index in matched_truths
            )

            if duplicate:
                duplicates += 1

            details.append({
                "status": "DUPLICATE" if duplicate else "FP",
                "confidence": pred["confidence"],
                "iou": best_iou,
                "bbox": pred["bbox"],
            })

    false_negatives = len(truths) - len(matched_truths)

    return {
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "duplicates": duplicates,
        "details": details,
        "ground_truth_count": len(truths),
        "prediction_count": len(preds),
    }


# ------------------------------------------------------------
# PRECISION / RECALL / F1
# ------------------------------------------------------------

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


# ------------------------------------------------------------
# AVERAGE PRECISION
# ------------------------------------------------------------

def calculate_ap(all_predictions, all_ground_truth, class_id, iou_threshold):
    total_truths = sum(
        sum(1 for g in item["ground_truth"] if g["class_id"] == class_id)
        for item in all_ground_truth
    )

    if total_truths == 0:
        return None

    ranked_predictions = []

    for image_index, predictions in enumerate(all_predictions):
        truths = [
            g for g in all_ground_truth[image_index]["ground_truth"]
            if g["class_id"] == class_id
        ]

        for pred in predictions:
            if pred["class_id"] == class_id:
                ranked_predictions.append((
                    pred["confidence"],
                    image_index,
                    pred,
                    truths,
                ))

    ranked_predictions.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    matched = {
        index: set()
        for index in range(len(all_ground_truth))
    }

    tp_values = []
    fp_values = []

    for _, image_index, pred, truths in ranked_predictions:
        best_iou = 0.0
        best_index = None

        for index, truth in enumerate(truths):
            if index in matched[image_index]:
                continue

            iou = calculate_iou(pred["bbox"], truth["bbox"])

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            matched[image_index].add(best_index)
            tp_values.append(1)
            fp_values.append(0)
        else:
            tp_values.append(0)
            fp_values.append(1)

    if not tp_values:
        return 0.0

    tp_cumulative = np.cumsum(tp_values)
    fp_cumulative = np.cumsum(fp_values)

    recalls = tp_cumulative / total_truths
    precisions = tp_cumulative / np.maximum(
        tp_cumulative + fp_cumulative, 1e-12
    )

    # 101-point interpolated AP.
    recall_levels = np.linspace(0, 1, 101)
    interpolated = []

    for recall_level in recall_levels:
        valid = precisions[recalls >= recall_level]
        interpolated.append(
            float(np.max(valid)) if len(valid) else 0.0
        )

    return float(np.mean(interpolated))


def select_balanced_images(image_paths, annotation_dir, limit, seed):
    """
    Select a reproducible, balanced sample across aircraft-count
    categories: 0, 1, 2, 3, 4, 5, 6, 7 and 8+.
    """

    if limit is None or limit >= len(image_paths):
        return sorted(image_paths)

    if limit <= 0:
        raise ValueError("--limit must be greater than zero.")

    categories = [0, 1, 2, 3, 4, 5, 6, 7, "8+"]

    # Allocate the sample as evenly as possible.
    base, remainder = divmod(limit, len(categories))
    quotas = {
        category: base + (1 if i < remainder else 0)
        for i, category in enumerate(categories)
    }

    buckets = {category: [] for category in categories}

    for image_path in image_paths:
        metadata_path = annotation_dir / f"{image_path.stem}.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing metadata required for balanced sampling: "
                f"{metadata_path}"
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        count = int(metadata["contact_count"])
        category = count if count <= 7 else "8+"
        buckets[category].append(image_path)

    rng = random.Random(seed)
    selected = []

    print("\nBALANCED SAMPLE DISTRIBUTION")

    for category in categories:
        candidates = sorted(buckets[category])
        required = quotas[category]

        if len(candidates) < required:
            raise ValueError(
                f"Category {category} has only {len(candidates)} images; "
                f"{required} required for this sample."
            )

        chosen = rng.sample(candidates, required)
        selected.extend(chosen)

        print(
            f"Category {str(category):<5}: "
            f"{len(chosen):>3} selected"
        )

    rng.shuffle(selected)

    print(f"Total selected: {len(selected)}")
    return selected

# ------------------------------------------------------------
# MAIN EVALUATION
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to trained YOLOv5 best.pt checkpoint",
    )

    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
    )

    parser.add_argument(
        "--repo",
        default=str(DEFAULT_YOLO_REPO),
    )

    parser.add_argument(
        "--image",
        default=None,
        help="Optionally evaluate one image, e.g. radar_00009",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of images to evaluate using balanced sampling.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261030,
        help="Random seed for reproducible image sampling.",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    image_dir = dataset / "images"
    label_dir = dataset / "annotations"

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            "Dataset images or annotations directory not found."
        )

    # Load all image paths first
    image_paths = sorted(image_dir.glob("*.png"))

    # Apply image selection
    if args.image:
        if args.limit is not None:
            raise ValueError(
                "Use either --image or --limit, not both."
            )

        image_name = Path(args.image).stem
        image_paths = [
            p for p in image_paths
            if p.stem == image_name
        ]

    elif args.limit is not None:
        image_paths = select_balanced_images(
            image_paths,
            label_dir,
            args.limit,
            args.seed,
        )

    if not image_paths:
        raise RuntimeError("No matching PNG images found.")
    model = load_model(Path(args.repo), Path(args.weights))
    print("\n========== YOLO MODEL DIAGNOSTICS ==========")
    print("Model class names:", model.names)
    print("Number of classes:", len(model.names))
    print("Dataset classes: 0 = aircraft, 1 = text")
    print("============================================\n")
    all_predictions = []
    all_ground_truth = []

    print(f"\nEvaluating {len(image_paths)} image(s)...")

    for index, image_path in enumerate(image_paths, start=1):
        import cv2

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        height, width = image.shape[:2]

        label_path = label_dir / f"{image_path.stem}.txt"

        truths = load_ground_truth(
            label_path,
            width,
            height,
        )

        predictions = predict_image(model, image_path)

        all_predictions.append(predictions)
        all_ground_truth.append({
            "image": image_path.name,
            "ground_truth": truths,
        })

        print(
            f"[{index}/{len(image_paths)}] {image_path.name} | "
            f"GT aircraft: {sum(g['class_id'] == 0 for g in truths)} | "
            f"Pred aircraft: {sum(p['class_id'] == 0 for p in predictions)}"
        )
    # --------------------------------------------------------
    # THRESHOLD COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("CONFIDENCE THRESHOLD EVALUATION")
    print("=" * 75)

    for class_id, class_name in CLASS_NAMES.items():
        print(f"\nCLASS: {class_name.upper()}")

        print(
            f"{'Threshold':>10} {'TP':>8} {'FP':>8} "
            f"{'FN':>8} {'Precision':>12} "
            f"{'Recall':>10} {'F1':>10}"
        )

        for threshold in CONFIDENCE_THRESHOLDS:
            totals = {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "duplicates": 0,
            }

            for predictions, item in zip(
                all_predictions,
                all_ground_truth,
            ):
                result = match_predictions(
                    predictions,
                    item["ground_truth"],
                    class_id,
                    threshold,
                    0.50,
                )

                for key in totals:
                    totals[key] += result[key]

            precision, recall, f1 = calculate_metrics(
                totals["TP"],
                totals["FP"],
                totals["FN"],
            )

            print(
                f"{threshold:>10.2f} "
                f"{totals['TP']:>8} "
                f"{totals['FP']:>8} "
                f"{totals['FN']:>8} "
                f"{precision:>11.4f} "
                f"{recall:>10.4f} "
                f"{f1:>10.4f}"
            )

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("MEAN AVERAGE PRECISION")
    print("=" * 75)

    for class_id, class_name in CLASS_NAMES.items():
        ap50 = calculate_ap(
            all_predictions,
            all_ground_truth,
            class_id,
            0.50,
        )

        aps = [
            calculate_ap(
                all_predictions,
                all_ground_truth,
                class_id,
                iou_threshold,
            )
            for iou_threshold in np.arange(0.50, 0.96, 0.05)
        ]

        aps = [value for value in aps if value is not None]
        map_5095 = float(np.mean(aps)) if aps else None

        print(f"\n{class_name}:")
        print(f"  AP@0.50:    {ap50:.4f}" if ap50 is not None else "  AP@0.50: N/A")
        print(f"  AP@0.50:0.95: {map_5095:.4f}" if map_5095 is not None else "  AP@0.50:0.95: N/A")

    # --------------------------------------------------------
    # INDIVIDUAL IMAGE REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("INDIVIDUAL IMAGE ERROR REPORT")
    print("=" * 75)

    for predictions, item in zip(
        all_predictions,
        all_ground_truth,
    ):
        result = match_predictions(
            predictions,
            item["ground_truth"],
            class_id=0,
            confidence_threshold=0.55,
            iou_threshold=0.50,
        )

        if result["FP"] or result["FN"] or result["duplicates"]:
            print(f"\n{item['image']}")
            print(
                f"GT={result['ground_truth_count']} "
                f"Pred={result['prediction_count']} "
                f"TP={result['TP']} FP={result['FP']} "
                f"FN={result['FN']} "
                f"Duplicates={result['duplicates']}"
            )

            for detail in result["details"]:
                if detail["status"] != "TP":
                    print(
                        f"  {detail['status']} | "
                        f"confidence={detail['confidence']:.4f} | "
                        f"IoU={detail['iou']:.4f} | "
                        f"box={detail['bbox']}"
                    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    report = {
        "images_evaluated": len(all_predictions),
        "confidence_thresholds": CONFIDENCE_THRESHOLDS,
        "iou_threshold": IOU_THRESHOLD,
        "results": [],
    }

    for predictions, item in zip(
        all_predictions,
        all_ground_truth,
    ):
        result = match_predictions(
            predictions,
            item["ground_truth"],
            class_id=0,
            confidence_threshold=0.55,
            iou_threshold=0.50,
        )

        report["results"].append({
            "image": item["image"],
            "ground_truth_aircraft": result["ground_truth_count"],
            "predicted_aircraft": result["prediction_count"],
            "TP": result["TP"],
            "FP": result["FP"],
            "FN": result["FN"],
            "duplicates": result["duplicates"],
            "details": result["details"],
        })

    output_path = PROJECT_ROOT / "yolo_evaluation_report.json"

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
    print("Evaluation completed.")


if __name__ == "__main__":
    main()
