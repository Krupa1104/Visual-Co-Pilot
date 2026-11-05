
import ast
import csv
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageStat

from src.dataset.ocr_crop_dataset import extract_callsign_crop

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data/synthetic_complete/images"
CSV_PATH = ROOT / "reports/yolo_ocr_100/results.csv"
OUTPUT_DIR = ROOT / "reports/crop_comparison"

IOU_MIN = 0.90
MAX_PAIRS = 20

def calculate_crop_difference(image, gt_box, pred_box):
    gt_crop = extract_callsign_crop(image, gt_box, padding=5)
    pred_crop = extract_callsign_crop(image, pred_box, padding=5)

    difference = ImageChops.difference(gt_crop, pred_crop)
    return sum(ImageStat.Stat(difference).mean) / 3

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    eligible = [
        row for row in rows
        if row["predicted_bbox"] not in ("", "None")
        and float(row["box_iou"]) >= IOU_MIN
    ]

    incorrect = [
        row for row in eligible
        if row["ocr_exact_match"].lower() == "false"
    ]

    correct = [
        row for row in eligible
        if row["ocr_exact_match"].lower() == "true"
    ]

    # Compare crop differences across ALL eligible predictions.
    correct_differences = []
    incorrect_differences = []

    for row in eligible:
        image_path = IMAGE_DIR / f"{row['image_id']}.png"
        image = Image.open(image_path).convert("RGB")

        gt_box = ast.literal_eval(row["gt_bbox"])
        pred_box = ast.literal_eval(row["predicted_bbox"])

        difference = calculate_crop_difference(
            image, gt_box, pred_box
        )

        if row["ocr_exact_match"].lower() == "true":
            correct_differences.append(difference)
        else:
            incorrect_differences.append(difference)

    print("\nALL HIGH-IoU CROP COMPARISONS")
    print("High-IoU eligible:", len(eligible))
    print("Correct predictions:", len(correct_differences))
    print("Incorrect predictions:", len(incorrect_differences))

    if correct_differences:
        print(
            "Mean difference - correct:",
            round(sum(correct_differences) / len(correct_differences), 3)
        )

    if incorrect_differences:
        print(
            "Mean difference - incorrect:",
            round(sum(incorrect_differences) / len(incorrect_differences), 3)
        )

    # Select incorrect and correct predictions for visual comparison.
    selected = incorrect[:MAX_PAIRS // 2] + correct[:MAX_PAIRS // 2]

    print("\nSELECTED CROP COMPARISONS")
    print("Pairs selected:", len(selected))

    report = []

    for index, row in enumerate(selected, start=1):
        image_id = row["image_id"]
        actual = row["actual_callsign"]
        predicted = row["predicted_callsign"]

        image_path = IMAGE_DIR / f"{image_id}.png"
        image = Image.open(image_path).convert("RGB")

        gt_box = ast.literal_eval(row["gt_bbox"])
        pred_box = ast.literal_eval(row["predicted_bbox"])

        gt_crop = extract_callsign_crop(image, gt_box, padding=5)
        pred_crop = extract_callsign_crop(image, pred_box, padding=5)

        # Pixel-level comparison after identical crop preprocessing.
        difference = ImageChops.difference(gt_crop, pred_crop)
        mean_difference = sum(ImageStat.Stat(difference).mean) / 3

        changed_pixels = sum(
            1 for pixel in difference.convert("L").getdata()
            if pixel > 0
        )
        changed_fraction = changed_pixels / (
            gt_crop.width * gt_crop.height
        )

        # Side-by-side comparison image.
        canvas = Image.new("RGB", (448, 125), "white")
        canvas.paste(gt_crop, (0, 25))
        canvas.paste(pred_crop, (224, 25))

        draw = ImageDraw.Draw(canvas)
        draw.text((5, 5), f"GT: {actual}", fill="black")
        draw.text((229, 5), f"YOLO: {predicted}", fill="black")

        output_path = OUTPUT_DIR / f"{index:02d}_{image_id}_{actual}.png"
        canvas.save(output_path)

        report.append({
            "image_id": image_id,
            "actual": actual,
            "predicted": predicted,
            "iou": float(row["box_iou"]),
            "ocr_correct": row["ocr_exact_match"],
            "mean_pixel_difference": round(mean_difference, 3),
            "changed_pixel_fraction": round(changed_fraction, 4),
            "comparison_image": output_path.name,
        })

        print(
            f"{index:02d}. {image_id} | {actual} -> {predicted} | "
            f"IoU={float(row['box_iou']):.3f} | "
            f"pixel difference={mean_difference:.2f}"
        )

    if report:
        with open(
            OUTPUT_DIR / "comparison_report.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=report[0].keys())
            writer.writeheader()
            writer.writerows(report)

    print("\nComparison images:", OUTPUT_DIR)
    print("Report:", OUTPUT_DIR / "comparison_report.csv")


if __name__ == "__main__":
    main()