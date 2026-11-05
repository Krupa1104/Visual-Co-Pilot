
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

QA_FILE = ROOT / "data" / "qa" / "annotations.jsonl"
IMAGE_DIR = ROOT / "data" / "synthetic" / "images"

EXPECTED_FRAMES = 4000
EXPECTED_QA_PER_IMAGE = 7
EXPECTED_TOTAL = EXPECTED_FRAMES * EXPECTED_QA_PER_IMAGE

REQUIRED_FIELDS = {
    "qa_id",
    "image_id",
    "image_path",
    "question_type",
    "question",
    "answer",
}

VALID_TYPES = {
    "callsign",
    "altitude",
    "heading",
    "aircraft_count",
    "yes_no",
    "quadrant",
}

VALID_ALTITUDES = {
    "10k-20k",
    "20k-30k",
    "30k-40k",
    "40k+",
}

VALID_HEADINGS = {
    "N", "NE", "E", "SE", "S", "SW", "W", "NW"
}

VALID_QUADRANTS = {
    "north-east",
    "north-west",
    "south-east",
    "south-west",
}


def main():
    print("=" * 60)
    print("RADAR QA DATASET VALIDATION")
    print("=" * 60)

    if not QA_FILE.is_file():
        raise FileNotFoundError(f"QA file not found: {QA_FILE}")

    records = []
    errors = []
    qa_ids = set()
    frame_counts = defaultdict(int)
    question_distribution = Counter()
    answer_distribution = defaultdict(Counter)

    with QA_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_number}: invalid JSON: {exc}")
                continue

            records.append(record)

            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                errors.append(
                    f"Line {line_number}: missing fields {sorted(missing)}"
                )
                continue

            qa_id = record["qa_id"]
            image_id = record["image_id"]
            question_type = record["question_type"]
            answer = record["answer"]

            if qa_id in qa_ids:
                errors.append(f"Duplicate QA ID: {qa_id}")
            qa_ids.add(qa_id)

            frame_counts[image_id] += 1
            question_distribution[question_type] += 1
            answer_distribution[question_type][answer] += 1

            if question_type not in VALID_TYPES:
                errors.append(
                    f"{qa_id}: unknown question type {question_type}"
                )

            if not str(record["question"]).strip():
                errors.append(f"{qa_id}: empty question")

            if not str(answer).strip():
                errors.append(f"{qa_id}: empty answer")

            if question_type == "altitude" and answer not in VALID_ALTITUDES:
                errors.append(f"{qa_id}: invalid altitude answer {answer}")

            if question_type == "heading" and answer not in VALID_HEADINGS:
                errors.append(f"{qa_id}: invalid heading answer {answer}")

            if question_type == "quadrant" and answer not in VALID_QUADRANTS:
                errors.append(f"{qa_id}: invalid quadrant answer {answer}")

            if question_type == "yes_no" and answer not in {"yes", "no"}:
                errors.append(f"{qa_id}: invalid yes/no answer {answer}")

            if question_type == "aircraft_count" and answer not in {
                "5", "6", "7", "8+"
            }:
                errors.append(f"{qa_id}: invalid aircraft count answer {answer}")

            expected_image = IMAGE_DIR / f"{image_id}.png"
            if not expected_image.is_file():
                errors.append(
                    f"{qa_id}: referenced image missing: {expected_image}"
                )

    # Check the total QA count.
    if len(records) != EXPECTED_TOTAL:
        errors.append(
            f"Expected {EXPECTED_TOTAL} records, found {len(records)}"
        )

    # Check each frame has exactly seven records.
    if len(frame_counts) != EXPECTED_FRAMES:
        errors.append(
            f"Expected {EXPECTED_FRAMES} unique frames, "
            f"found {len(frame_counts)}"
        )

    for image_id, count in frame_counts.items():
        if count != EXPECTED_QA_PER_IMAGE:
            errors.append(
                f"{image_id}: expected 7 QA pairs, found {count}"
            )

    print(f"\nQA records       : {len(records):,}")
    print(f"Unique QA IDs    : {len(qa_ids):,}")
    print(f"Unique images    : {len(frame_counts):,}")

    print("\nQuestion-type distribution:")
    for question_type, count in sorted(question_distribution.items()):
        print(f"  {question_type:<20} {count:>8,}")

    print("\nAnswer distributions:")
    for question_type, answers in sorted(answer_distribution.items()):
        print(f"\n{question_type}:")
        for answer, count in sorted(answers.items()):
            print(f"  {answer:<20} {count:>8,}")

    print(f"\nValidation errors: {len(errors)}")

    for error in errors[:30]:
        print("ERROR:", error)

    if len(errors) > 30:
        print(f"... and {len(errors) - 30} more errors")

    if errors:
        print("\nQA VALIDATION FAILED")
    else:
        print("\nQA VALIDATION PASSED")

    print("=" * 60)


if __name__ == "__main__":
    main()
