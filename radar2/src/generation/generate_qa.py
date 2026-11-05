
"""
Fresh synthetic radar QA-pair generator.

Input:
    data/synthetic/all_frames.json
    data/synthetic/images/

Output:
    data/qa/annotations.jsonl

Run:
    python src/generation/generate_qa.py

Regenerate:
    python src/generation/generate_qa.py --overwrite
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = ROOT / "data" / "synthetic" / "all_frames.json"
IMAGE_DIR = ROOT / "data" / "synthetic" / "images"
OUTPUT_PATH = ROOT / "data" / "qa" / "annotations.jsonl"

RANDOM_SEED = 20261030

# Four altitude classes, covering the generator's 10,000–49,900 ft range.
ALTITUDE_BUCKETS = [
    (10000, 20000, "10k-20k"),
    (20000, 30000, "20k-30k"),
    (30000, 40000, "30k-40k"),
    (40000, 50000, "40k+"),
]

# Eight compass directions.
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Nine aircraft-count classes: 0, 1, 2, 3, 4, 5, 6, 7, 8+.
def count_category(count):
    return str(count) if count <= 7 else "8+"


def get_altitude_category(altitude):
    for lower, upper, label in ALTITUDE_BUCKETS:
        if lower <= altitude < upper:
            return label

    raise ValueError(f"Unsupported altitude: {altitude}")


def get_heading_category(heading):
    """
    Convert heading degrees to one of eight compass directions.
    0 degrees = North; angles increase clockwise.
    """
    index = int(((heading % 360) + 22.5) // 45) % 8
    return DIRECTIONS[index]


def get_quadrant(position, width, height):
    """
    Determine the screen quadrant using aircraft pixel coordinates.
    """
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
        "answer": str(answer)
    }


def generate_frame_qa(frame, rng):
    image_name = frame["image"]
    frame_id = Path(image_name).stem

    image_path = IMAGE_DIR / image_name

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image missing for frame {frame_id}: {image_path}"
        )

    contacts = frame["contacts"]

    if not contacts:
        raise ValueError(f"No aircraft contacts in {frame_id}")

    width = frame["width"]
    height = frame["height"]
    aircraft_count = frame["contact_count"]

    if aircraft_count != len(contacts):
        raise ValueError(
            f"{frame_id}: contact_count does not match contacts list"
        )

    # Randomly select aircraft for individual questions.
    shuffled = contacts.copy()
    rng.shuffle(shuffled)

    first = shuffled[0]
    second = shuffled[1] if len(shuffled) > 1 else shuffled[0]

    altitude_contact = rng.choice(contacts)
    heading_contact = rng.choice(contacts)
    quadrant_contact = rng.choice(contacts)

    # Convert aircraft headings to compass sectors.
    aircraft_directions = {
        get_heading_category(int(contact["heading"]))
        for contact in contacts
    }

    # Choose a direction and ask whether any aircraft is heading that way.
    # This produces both YES and NO answers across the dataset.
    queried_direction = rng.choice(DIRECTIONS)
    direction_present = queried_direction in aircraft_directions

    pairs = []

    # 1. Callsign
    pairs.append(make_record(
        frame_id,
        image_path,
        "callsign",
        "What is the callsign of the selected aircraft?",
        first["callsign"],
        1
    ))

    # 2. Altitude bucket
    altitude = int(altitude_contact["altitude"])

    pairs.append(make_record(
        frame_id,
        image_path,
        "altitude",
        f"Which altitude category contains aircraft "
        f"{altitude_contact['callsign']}?",
        get_altitude_category(altitude),
        2
    ))

    # 3. Compass heading
    heading = int(heading_contact["heading"])

    pairs.append(make_record(
        frame_id,
        image_path,
        "heading",
        f"What compass direction is aircraft "
        f"{heading_contact['callsign']} heading?",
        get_heading_category(heading),
        3
    ))

    # 4. Aircraft count
    pairs.append(make_record(
        frame_id,
        image_path,
        "aircraft_count",
        "How many aircraft are present in the radar image?",
        count_category(aircraft_count),
        4
    ))

    # 5. Yes / No
    pairs.append(make_record(
        frame_id,
        image_path,
        "yes_no",
        f"Is any aircraft heading {queried_direction}?",
        "yes" if direction_present else "no",
        5
    ))

    # 6. Screen quadrant
    pairs.append(make_record(
        frame_id,
        image_path,
        "quadrant",
        f"Which quadrant contains aircraft "
        f"{quadrant_contact['callsign']}?",
        get_quadrant(
            quadrant_contact["position"],
            width,
            height
        ),
        6
    ))

    # 7. Another callsign
    pairs.append(make_record(
        frame_id,
        image_path,
        "callsign",
        "What is the callsign of another aircraft in the image?",
        second["callsign"],
        7
    ))

    return pairs


def load_manifest():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        frames = json.load(file)

    if not isinstance(frames, list):
        raise ValueError("Expected manifest to be a JSON list.")

    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Generate fresh radar QA pairs."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing QA output file."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED
    )

    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Use --overwrite to regenerate it."
        )

    frames = load_manifest()
    rng = random.Random(args.seed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    distribution = Counter()
    total_pairs = 0
    seen_frame_ids = set()

    print("=" * 65)
    print("FRESH RADAR QA GENERATION")
    print("=" * 65)
    print(f"Frames         : {len(frames):,}")
    print("QA per image   : 7")
    print(f"Expected pairs : {len(frames) * 7:,}")
    print(f"Output         : {OUTPUT_PATH}")
    print()

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as output:

        for index, frame in enumerate(frames):

            frame_id = Path(frame["image"]).stem

            if frame_id in seen_frame_ids:
                raise ValueError(
                    f"Duplicate frame ID: {frame_id}"
                )

            seen_frame_ids.add(frame_id)

            pairs = generate_frame_qa(frame, rng)

            if len(pairs) != 7:
                raise RuntimeError(
                    f"{frame_id}: expected 7 pairs, got {len(pairs)}"
                )

            for pair in pairs:
                output.write(
                    json.dumps(pair, ensure_ascii=False) + "\n"
                )

                distribution[pair["question_type"]] += 1
                total_pairs += 1

            if (index + 1) % 250 == 0 or index + 1 == len(frames):
                print(
                    f"Processed {index + 1:>5}/{len(frames)} frames"
                )

    expected = len(frames) * 7

    if total_pairs != expected:
        raise RuntimeError(
            f"Expected {expected} QA pairs, got {total_pairs}"
        )

    print("\n" + "=" * 65)
    print("QA GENERATION COMPLETED")
    print("=" * 65)

    print(f"Frames processed : {len(seen_frame_ids):,}")
    print(f"QA pairs created : {total_pairs:,}")

    print("\nQuestion type distribution:")

    for question_type, count in sorted(distribution.items()):
        print(f"  {question_type:<20} {count:>8,}")

    print(f"\nSaved to: {OUTPUT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
