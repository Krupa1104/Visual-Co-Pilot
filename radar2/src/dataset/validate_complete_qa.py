
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = ROOT / "data" / "synthetic_complete" / "all_frames.json"
IMAGE_DIR = ROOT / "data" / "synthetic_complete" / "images"
QA_PATH = ROOT / "data" / "qa_complete" / "annotations.jsonl"

EXPECTED_FRAMES = 4500
EXPECTED_PAIRS_PER_FRAME = 7

VALID_TYPES = {
    "callsign", "altitude", "heading",
    "aircraft_count", "yes_no", "quadrant"
}
VALID_ALTITUDES = {"10k-20k", "20k-30k", "30k-40k", "40k+"}
VALID_HEADINGS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
VALID_COUNTS = {str(i) for i in range(8)} | {"8+"}
VALID_QUADRANTS = {
    "north-east", "north-west", "south-east", "south-west"
}


def fail(message):
    raise ValueError(message)


def main():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        frames = json.load(f)

    if len(frames) != EXPECTED_FRAMES:
        fail(f"Expected {EXPECTED_FRAMES} frames, found {len(frames)}")

    frame_counts = {
        Path(frame["image"]).stem: frame["contact_count"]
        for frame in frames
    }

    if len(frame_counts) != EXPECTED_FRAMES:
        fail("Duplicate frame IDs in manifest")

    if not QA_PATH.is_file():
        fail(f"QA file not found: {QA_PATH}")

    records = []
    qa_ids = set()
    pairs_by_frame = defaultdict(list)
    type_counts = Counter()
    answer_counts = Counter()
    errors = []

    with QA_PATH.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_number}: invalid JSON: {exc}")
                continue

            required = {
                "qa_id", "image_id", "image_path",
                "question_type", "question", "answer"
            }

            if not required.issubset(record):
                errors.append(f"Line {line_number}: missing required fields")
                continue

            qa_id = record["qa_id"]
            frame_id = record["image_id"]
            qtype = record["question_type"]
            answer = record["answer"]

            if qa_id in qa_ids:
                errors.append(f"Duplicate QA ID: {qa_id}")
            qa_ids.add(qa_id)

            if frame_id not in frame_counts:
                errors.append(f"Unknown image ID: {frame_id}")
                continue

            if qtype not in VALID_TYPES:
                errors.append(f"{qa_id}: invalid question type {qtype}")

            expected_path = (
                Path("data") / "synthetic_complete" /
                "images" / f"{frame_id}.png"
            )

            if Path(record["image_path"]) != expected_path:
                errors.append(f"{qa_id}: incorrect image path")

            if not (ROOT / record["image_path"]).is_file():
                errors.append(f"{qa_id}: referenced image does not exist")

            if not isinstance(record["question"], str) or not record["question"].strip():
                errors.append(f"{qa_id}: empty question")

            if not isinstance(answer, str) or not answer.strip():
                errors.append(f"{qa_id}: empty answer")

            if qtype == "aircraft_count" and answer not in VALID_COUNTS:
                errors.append(f"{qa_id}: invalid aircraft-count answer")

            if qtype == "altitude" and answer not in VALID_ALTITUDES:
                errors.append(f"{qa_id}: invalid altitude answer")

            if qtype == "heading" and answer not in VALID_HEADINGS:
                errors.append(f"{qa_id}: invalid heading answer")

            if qtype == "quadrant" and answer not in VALID_QUADRANTS:
                errors.append(f"{qa_id}: invalid quadrant answer")

            if qtype == "yes_no" and answer not in {"yes", "no"}:
                errors.append(f"{qa_id}: invalid yes/no answer")

            pairs_by_frame[frame_id].append(record)
            type_counts[qtype] += 1
            answer_counts[(qtype, answer)] += 1
            records.append(record)

    # Every manifest image must have exactly seven QA pairs.
    for frame_id, count in frame_counts.items():
        pairs = pairs_by_frame.get(frame_id, [])

        if len(pairs) != EXPECTED_PAIRS_PER_FRAME:
            errors.append(
                f"{frame_id}: expected 7 QA pairs, found {len(pairs)}"
            )

        if count == 0:
            if sum(p["question_type"] == "aircraft_count" for p in pairs) != 1:
                errors.append(f"{frame_id}: zero-aircraft count QA missing")

            if any(
                p["question_type"] in {
                    "callsign", "altitude", "heading", "quadrant"
                }
                for p in pairs
            ):
                errors.append(
                    f"{frame_id}: invalid aircraft-specific QA for empty scene"
                )

            if any(
                p["question_type"] == "yes_no" and p["answer"] != "no"
                for p in pairs
            ):
                errors.append(
                    f"{frame_id}: zero-aircraft yes/no answer must be no"
                )

            if any(
                p["question_type"] == "aircraft_count" and p["answer"] != "0"
                for p in pairs
            ):
                errors.append(
                    f"{frame_id}: zero-aircraft count answer must be 0"
                )

    expected_total = EXPECTED_FRAMES * EXPECTED_PAIRS_PER_FRAME

    print("=" * 65)
    print("COMPLETE QA DATASET VALIDATION")
    print("=" * 65)
    print(f"Manifest frames : {len(frames):,}")
    print(f"QA records      : {len(records):,}")
    print(f"Unique QA IDs   : {len(qa_ids):,}")
    print(f"Frames with QA  : {len(pairs_by_frame):,}")

    print("\nQuestion-type distribution:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype:<20} {count:>8,}")

    print("\nAircraft-count answer distribution:")
    for answer in sorted(VALID_COUNTS, key=lambda x: (x == "8+", int(x) if x != "8+" else 8)):
        print(f"  {answer:<8} {answer_counts[('aircraft_count', answer)]:>8,}")

    print(f"\nValidation errors: {len(errors)}")

    if errors:
        for error in errors[:50]:
            print("ERROR:", error)
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more errors")
        raise SystemExit("QA VALIDATION FAILED")

    if len(records) != expected_total:
        fail(f"Expected {expected_total} records, found {len(records)}")

    print("\nCOMPLETE QA VALIDATION PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
