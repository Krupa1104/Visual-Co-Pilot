"""
consolidate_dataset.py
Merges all 5 phase folders (engine_off, taxi, takeoff_roll, climb, level_flight)
into one unified dataset: a single images folder, one master CSV, and a
shuffled train/val/test split.

Run this after all phase captures are complete.
"""

import os
import csv
import shutil
import random

# === CONFIG ===
BASE_DIR = r"C:\Users\krupa\sem6\capstone\datasets\asritha"
PHASES = ["taxi", "takeoff_roll", "climb", "level_flight", "engine_off"]
# NOTE: engine_off here should point to the folder containing the NEW
# 150 synthetic frames only. If your synthetic engine_off run overwrote
# the original 40-frame folder, this is already handled — nothing to do.
# If the original 40 frames are still mixed in, move/delete them first.

OUTPUT_DIR = os.path.join(BASE_DIR, "consolidated")
IMAGES_OUT = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGES_OUT, exist_ok=True)

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# === MERGE CSVs + COPY IMAGES ===
all_rows = []
fieldnames = None

for phase in PHASES:
    csv_path = os.path.join(BASE_DIR, phase, "flight_log.csv")
    images_dir = os.path.join(BASE_DIR, phase, "images")

    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found — skipping phase '{phase}'")
        continue

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"{phase}: {len(rows)} rows")

    copied = 0
    for row in rows:
        src = os.path.join(images_dir, row["image"])
        dst = os.path.join(IMAGES_OUT, row["image"])
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  WARNING: image not found, skipping row: {src}")
            continue
        all_rows.append(row)

    print(f"  copied {copied} images")

print(f"\nTotal consolidated rows: {len(all_rows)}")

# === SHUFFLE + SPLIT (not sequential — avoids biasing splits by phase order
# or by monotonically-ramping altitude within a phase) ===
random.seed(RANDOM_SEED)
random.shuffle(all_rows)

n = len(all_rows)
n_train = int(n * SPLIT_RATIOS["train"])
n_val = int(n * SPLIT_RATIOS["val"])

train_rows = all_rows[:n_train]
val_rows = all_rows[n_train:n_train + n_val]
test_rows = all_rows[n_train + n_val:]

print(f"Split -> train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)}")

# === WRITE MASTER CSV + SPLIT CSVs ===
master_csv = os.path.join(OUTPUT_DIR, "master_flight_log.csv")
with open(master_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nWrote master CSV: {master_csv}")

for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
    split_csv = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
    with open(split_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {split_name} split: {split_csv} ({len(rows)} rows)")

# === PER-PHASE COUNT SUMMARY (sanity check the split isn't accidentally
# skewed toward one phase) ===
print("\nPhase distribution per split:")
for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
    counts = {}
    for r in rows:
        counts[r["phase"]] = counts.get(r["phase"], 0) + 1
    print(f"  {split_name}: {counts}")

print(f"\nAll images consolidated into: {IMAGES_OUT}")
print("Done.")
