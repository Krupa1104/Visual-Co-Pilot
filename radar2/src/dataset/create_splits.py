
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = ROOT / "data" / "synthetic_complete" / "all_frames.json"
QA_PATH = ROOT / "data" / "qa_complete" / "annotations.jsonl"
OUTPUT_DIR = ROOT / "data" / "splits"

SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def main():
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.")

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} is not empty. Refusing to overwrite existing splits."
        )

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        frames = json.load(f)

    with QA_PATH.open("r", encoding="utf-8") as f:
        qa_records = [json.loads(line) for line in f if line.strip()]

    frame_ids = [Path(frame["image"]).stem for frame in frames]

    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("Duplicate frame IDs found.")

    qa_by_frame = {}
    for record in qa_records:
        qa_by_frame.setdefault(record["image_id"], []).append(record)

    if set(qa_by_frame) != set(frame_ids):
        raise ValueError("QA frame IDs do not match manifest frame IDs.")

    if any(len(records) != 7 for records in qa_by_frame.values()):
        raise ValueError("Every frame must have exactly 7 QA records.")

    rng = random.Random(SEED)
    shuffled_ids = frame_ids.copy()
    rng.shuffle(shuffled_ids)

    total = len(shuffled_ids)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    split_ids = {
        "train": shuffled_ids[:train_end],
        "val": shuffled_ids[train_end:val_end],
        "test": shuffled_ids[val_end:],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, ids in split_ids.items():
        records = [
            record
            for frame_id in ids
            for record in qa_by_frame[frame_id]
        ]

        with (OUTPUT_DIR / f"{split_name}.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with (OUTPUT_DIR / f"{split_name}_frames.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(ids, f, indent=2)

    # Verify no frame leakage.
    train = set(split_ids["train"])
    val = set(split_ids["val"])
    test = set(split_ids["test"])

    assert not (train & val or train & test or val & test)
    assert train | val | test == set(frame_ids)

    print("=" * 55)
    print("DATASET SPLIT COMPLETED")
    print("=" * 55)

    for name, ids in split_ids.items():
        print(
            f"{name:<8}: {len(ids):,} images | "
            f"{len(ids) * 7:,} QA pairs"
        )

    print("\nAircraft-count distribution by split:")
    manifest_counts = {
        Path(frame["image"]).stem: frame["contact_count"]
        for frame in frames
    }

    for name, ids in split_ids.items():
        counts = Counter(
            "8+" if manifest_counts[i] >= 8
            else str(manifest_counts[i])
            for i in ids
        )
        print(f"\n{name}:")
        for category in ["0", "1", "2", "3", "4", "5", "6", "7", "8+"]:
            print(f"  {category:<4}: {counts[category]}")

    print(f"\nSaved splits to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
