
"""
QA generator for the new balanced 4,500-image radar dataset.

Input:
    data/synthetic_complete/all_frames.json
    data/synthetic_complete/images/

Output:
    data/qa_complete/annotations.jsonl

Run:
    python src/generation/generate_complete_qa.py
"""

import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = ROOT / "data" / "synthetic_complete" / "all_frames.json"
IMAGE_DIR = ROOT / "data" / "synthetic_complete" / "images"
OUTPUT_PATH = ROOT / "data" / "qa_complete" / "annotations.jsonl"

RANDOM_SEED = 20261031

ALTITUDE_BUCKETS = [
    (10000, 20000, "10k-20k"),
    (20000, 30000, "20k-30k"),
    (30000, 40000, "30k-40k"),
    (40000, 50000, "40k+"),
]

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

QUADRANTS = [
    "north-west",
    "north-east",
    "south-west",
    "south-east",
]


def count_category(count):
    return str(count) if count <= 7 else "8+"


def get_altitude_category(altitude):
    for lower, upper, label in ALTITUDE_BUCKETS:
        if lower <= altitude < upper:
            return label
    raise ValueError(f"Unsupported altitude: {altitude}")


def get_heading_category(heading):
    index = int(((heading % 360) + 22.5) // 45) % 8
    return DIRECTIONS[index]


def get_quadrant(position, width, height):
    x, y = position

    east_west = "east" if x >= width / 2 else "west"
    north_south = "south" if y >= height / 2 else "north"

    return f"{north_south}-{east_west}"


def make_record(frame_id, image_path, question_type,
                question, answer, pair_number):

    return {
        "qa_id": f"{frame_id}_qa_{pair_number:02d}",
        "image_id": frame_id,
        "image_path": str(image_path.relative_to(ROOT)),
        "question_type": question_type,
        "question": question,
        "answer": str(answer),
    }


def validate_frame(frame):
    required = {
        "image", "width", "height", "contact_count", "contacts"
    }

    missing = required - frame.keys()
    if missing:
        raise ValueError(f"Manifest frame missing fields: {sorted(missing)}")

    if not isinstance(frame["contacts"], list):
        raise ValueError(f"{frame['image']}: contacts must be a list")

    if frame["contact_count"] != len(frame["contacts"]):
        raise ValueError(
            f"{frame['image']}: contact_count does not match contacts"
        )


def generate_zero_aircraft_qa(frame, image_path):
    frame_id = Path(frame["image"]).stem
    pairs = []

    # 1. Aircraft count
    pairs.append(make_record(
        frame_id, image_path, "aircraft_count",
        "How many aircraft are present in the radar image?",
        "0", 1
    ))

    # 2–5. No aircraft can be heading in any of these directions.
    for pair_number, direction in enumerate(
        ["N", "E", "S", "W"], start=2
    ):
        pairs.append(make_record(
            frame_id, image_path, "yes_no",
            f"Is any aircraft heading {direction}?",
            "no", pair_number
        ))

    # 6–7. No aircraft can occupy either queried quadrant.
    for pair_number, quadrant in zip(
        [6, 7], ["north-west", "south-east"]
    ):
        pairs.append(make_record(
            frame_id, image_path, "yes_no",
            f"Is there any aircraft in the {quadrant} quadrant?",
            "no", pair_number
        ))

    return pairs


def generate_nonzero_aircraft_qa(frame, image_path, rng):
    frame_id = Path(frame["image"]).stem
    contacts = frame["contacts"]

    width = frame["width"]
    height = frame["height"]
    aircraft_count = frame["contact_count"]

    shuffled = contacts.copy()
    rng.shuffle(shuffled)

    first = shuffled[0]
    second = shuffled[1] if len(shuffled) > 1 else shuffled[0]

    altitude_contact = rng.choice(contacts)
    heading_contact = rng.choice(contacts)
    quadrant_contact = rng.choice(contacts)

    aircraft_directions = {
        get_heading_category(int(contact["heading"]))
        for contact in contacts
    }

    queried_direction = rng.choice(DIRECTIONS)
    direction_present = queried_direction in aircraft_directions

    altitude = int(altitude_contact["altitude"])
    heading = int(heading_contact["heading"])

    return [
        make_record(
            frame_id, image_path, "callsign",
            "What is the callsign of the selected aircraft?",
            first["callsign"], 1
        ),
        make_record(
            frame_id, image_path, "altitude",
            f"Which altitude category contains aircraft "
            f"{altitude_contact['callsign']}?",
            get_altitude_category(altitude), 2
        ),
        make_record(
            frame_id, image_path, "heading",
            f"What compass direction is aircraft "
            f"{heading_contact['callsign']} heading?",
            get_heading_category(heading), 3
        ),
        make_record(
            frame_id, image_path, "aircraft_count",
            "How many aircraft are present in the radar image?",
            count_category(aircraft_count), 4
        ),
        make_record(
            frame_id, image_path, "yes_no",
            f"Is any aircraft heading {queried_direction}?",
            "yes" if direction_present else "no", 5
        ),
        make_record(
            frame_id, image_path, "quadrant",
            f"Which quadrant contains aircraft "
            f"{quadrant_contact['callsign']}?",
            get_quadrant(
                quadrant_contact["position"], width, height
            ), 6
        ),
        make_record(
            frame_id, image_path, "callsign",
            "What is the callsign of another aircraft in the image?",
            second["callsign"], 7
        ),
    ]


def generate_frame_qa(frame, rng):
    validate_frame(frame)

    image_name = frame["image"]
    image_path = IMAGE_DIR / image_name

    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")

    if frame["contact_count"] == 0:
        return generate_zero_aircraft_qa(frame, image_path)

    return generate_nonzero_aircraft_qa(frame, image_path, rng)


def load_manifest():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        frames = json.load(file)

    if not isinstance(frames, list):
        raise ValueError("Expected manifest to be a JSON list.")

    return frames


def main():
    frames = load_manifest()

    if len(frames) != 4500:
        raise ValueError(
            f"Expected 4500 frames, found {len(frames)}"
        )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}. "
            "It will not be overwritten."
        )

    rng = random.Random(RANDOM_SEED)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = OUTPUT_PATH.with_name(
        OUTPUT_PATH.name + ".tmp"
    )

    if temporary_path.exists():
        raise FileExistsError(
            f"Temporary output already exists: {temporary_path}"
        )

    distribution = Counter()
    answer_distribution = Counter()
    seen_frame_ids = set()
    total_pairs = 0

    try:
        with temporary_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as output:

            for index, frame in enumerate(frames):
                frame_id = Path(frame["image"]).stem

                if frame_id in seen_frame_ids:
                    raise ValueError(f"Duplicate frame ID: {frame_id}")

                seen_frame_ids.add(frame_id)

                pairs = generate_frame_qa(frame, rng)

                if len(pairs) != 7:
                    raise RuntimeError(
                        f"{frame_id}: expected 7 pairs, got {len(pairs)}"
                    )

                if len({pair["qa_id"] for pair in pairs}) != 7:
                    raise RuntimeError(f"{frame_id}: duplicate QA IDs")

                for pair in pairs:
                    output.write(
                        json.dumps(pair, ensure_ascii=False) + "\n"
                    )
                    distribution[pair["question_type"]] += 1
                    answer_distribution[
                        (pair["question_type"], pair["answer"])
                    ] += 1
                    total_pairs += 1

                if (index + 1) % 500 == 0:
                    print(f"Processed {index + 1}/4500 frames")

        expected_pairs = len(frames) * 7

        if total_pairs != expected_pairs:
            raise RuntimeError(
                f"Expected {expected_pairs} pairs, got {total_pairs}"
            )

        temporary_path.replace(OUTPUT_PATH)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    print("\n" + "=" * 65)
    print("COMPLETE RADAR QA GENERATION FINISHED")
    print("=" * 65)
    print(f"Frames processed : {len(seen_frame_ids):,}")
    print(f"QA pairs created : {total_pairs:,}")

    print("\nQuestion type distribution:")
    for question_type, count in sorted(distribution.items()):
        print(f"  {question_type:<20} {count:>8,}")

    print("\nAnswer distribution:")
    for (question_type, answer), count in sorted(answer_distribution.items()):
        print(f"  {question_type:<16} {answer:<16} {count:>7,}")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
