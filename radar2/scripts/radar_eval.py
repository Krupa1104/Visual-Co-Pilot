r"""
radar_eval.py -- accuracy and time per question type for the radar module, on TEST images only.

Test images = data/splits/test_frames.json from the radar team's create_splits.py (seed 42, 80/10/10 by
image, 450 test images). Images that are also in YOLO's own train/ or val/ folders
(data/yolo_dataset/images/...) are dropped, so neither the VQA model nor YOLO has seen them.
Questions and ground-truth answers come from the split's test.jsonl (7 per image, 6 types).

For each question it calls RadarSystem.handle_question exactly as the app does, and scores the answer:
  aircraft_count  first number in the answer (ground truth "8+" = 8 or more)
  callsign        "strict": the answer names exactly one callsign and it is the right one;
                  "lenient": the right callsign appears anywhere (an answer listing ALL callsigns
                  passes lenient but not strict). The headline uses strict.
  altitude        feet in the answer mapped to 10k-20k / 20k-30k / 30k-40k / 40k+, or the category text
  heading         degrees in the answer mapped to N/NE/E/SE/S/SW/W/NW, or the compass word
  yes_no          answer starts with yes / no
  quadrant        compass quadrant in the answer (top-right is counted as north-east, etc.)
It also records which route answered (the verdict), so routing problems show up per type.

Run from the repo root (venv_final, plugged in):
  python radar2\scripts\radar_eval.py --n 20 --seed 2                       (app default: mobile + crops)
  python radar2\scripts\radar_eval.py --n 20 --seed 2 --ocr_models medium     (accuracy mode)
  python radar2\scripts\radar_eval.py --n 10 --ocr_models medium --ocr_mode full   (radar team's original OCR)
Output: radar2/outputs/radar_eval/<tag>/per_question.csv and summary.json
"""
import argparse
import collections
import csv
import json
import os
import random
import re
import statistics
import sys
import time

import torch  # noqa: F401  -- before paddle on Windows

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
from radar_system import RadarSystem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default=os.path.join(BASE, "data", "splits"),
                help="folder with test_frames.json and test.jsonl (default: radar2/data/splits)")
ap.add_argument("--n", type=int, default=10, help="questions per type")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ocr_models", default="mobile", choices=["mobile", "medium"])
ap.add_argument("--ocr_mode", default="crops", choices=["crops", "full"])
args = ap.parse_args()
tag = f"{args.ocr_models}_{args.ocr_mode}_n{args.n}_seed{args.seed}"

# ---------------------------------------------------------------- test questions
test_frames = set(json.load(open(os.path.join(args.splits, "test_frames.json"), encoding="utf-8")))
yolo_seen = set()
for sub in ("train", "val"):
    d = os.path.join(BASE, "data", "yolo_dataset", "images", sub)
    if os.path.isdir(d):
        yolo_seen |= {os.path.splitext(f)[0] for f in os.listdir(d)}
usable = test_frames - yolo_seen
print(f"test images: {len(test_frames)}; also in YOLO train/val: {len(test_frames & yolo_seen)}; "
      f"usable: {len(usable)}")

by_type = collections.defaultdict(list)
with open(os.path.join(args.splits, "test.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["image_id"] in usable:
            by_type[r["question_type"]].append(r)
rng = random.Random(args.seed)
sample = []
for t in sorted(by_type):
    rows = sorted(by_type[t], key=lambda r: r["qa_id"])
    rng.shuffle(rows)
    sample += rows[:args.n]

# ---------------------------------------------------------------- scoring
COMPASS8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
QUAD = {"north-east": ("north-east", "northeast", "top right", "top-right", "upper right"),
        "north-west": ("north-west", "northwest", "top left", "top-left", "upper left"),
        "south-east": ("south-east", "southeast", "bottom right", "bottom-right", "lower right"),
        "south-west": ("south-west", "southwest", "bottom left", "bottom-left", "lower left")}
CALLSIGN = re.compile(r"\b[A-Z]{2,5}\d{1,5}[A-Z]?\b")


def alt_category(ft):
    return "10k-20k" if ft < 20000 else "20k-30k" if ft < 30000 else "30k-40k" if ft < 40000 else "40k+"


def score(qtype, gt, ans):
    a = ans or ""
    low = a.lower()
    if qtype == "aircraft_count":
        m = re.search(r"\b(\d+)\b", a)
        if not m:
            return False
        n = int(m.group(1))
        return n >= 8 if gt == "8+" else n == int(gt)
    if qtype == "callsign":
        return set(CALLSIGN.findall(a.upper())) == {gt.upper()}
    if qtype == "altitude":
        m = re.search(r"(\d{1,3}(?:,\d{3})+|\d{4,6})\s*(?:ft|feet)", low)
        if m:
            return alt_category(int(m.group(1).replace(",", ""))) == gt
        return gt.lower() in low
    if qtype == "heading":
        m = re.search(r"(\d{1,3})\s*°|heading of (\d{1,3})", a)
        if m:
            deg = int(m.group(1) or m.group(2)) % 360
            return COMPASS8[int((deg + 22.5) // 45) % 8] == gt
        return re.search(rf"\b{gt}\b", a) is not None
    if qtype == "yes_no":
        return low.strip().startswith(gt)
    if qtype == "quadrant":
        return any(p in low for p in QUAD.get(gt, (gt,)))
    return gt.lower() in low


# ---------------------------------------------------------------- run
print(f"Loading RadarSystem (ocr_models={args.ocr_models}, ocr_mode={args.ocr_mode}) ...")
rs = RadarSystem(base_dir=BASE, ocr_models=args.ocr_models, ocr_mode=args.ocr_mode)
warm = sample[0]
rs.handle_question(os.path.join(BASE, warm["image_path"]), "What is the callsign of the selected aircraft?")

out = []
for i, r in enumerate(sample, 1):
    img = os.path.join(BASE, r["image_path"])
    t0 = time.perf_counter()
    res = rs.handle_question(img, r["question"])
    secs = time.perf_counter() - t0
    ans = res.get("answer", "")
    ok = score(r["question_type"], r["answer"], ans)
    row = {"qa_id": r["qa_id"], "type": r["question_type"], "question": r["question"],
           "expected": r["answer"], "answer": ans, "verdict": res.get("verdict"), "correct": ok,
           "seconds": round(secs, 3)}
    if r["question_type"] == "callsign":
        row["correct_lenient"] = r["answer"].upper() in ans.upper()
    out.append(row)
    print(f"[{i}/{len(sample)}] {r['question_type']:<14} {'OK ' if ok else 'BAD'} {secs:6.1f} s "
          f"{res.get('verdict')}: {r['question']!r} -> {ans[:90]!r} (expected {r['answer']})", flush=True)

summary = {"setup": {"ocr_models": args.ocr_models, "ocr_mode": args.ocr_mode, "n_per_type": args.n,
                     "seed": args.seed, "usable_test_images": len(usable),
                     "test_images_also_in_yolo_train_val": len(test_frames & yolo_seen)}}
for t in sorted({o["type"] for o in out}):
    rows = [o for o in out if o["type"] == t]
    secs = [o["seconds"] for o in rows]
    s = {"correct": sum(o["correct"] for o in rows), "of": len(rows),
         "median_s": round(statistics.median(secs), 2), "max_s": round(max(secs), 2),
         "verdicts": dict(collections.Counter(o["verdict"] for o in rows))}
    if t == "callsign":
        s["correct_lenient"] = sum(o["correct_lenient"] for o in rows)
    summary[t] = s
summary["all"] = {"correct": sum(o["correct"] for o in out), "of": len(out),
                  "median_s": round(statistics.median(o["seconds"] for o in out), 2)}

od = os.path.join(BASE, "outputs", "radar_eval", tag)
os.makedirs(od, exist_ok=True)
with open(os.path.join(od, "per_question.csv"), "w", newline="", encoding="utf-8") as f:
    fields = list(dict.fromkeys(k for o in out for k in o))
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(out)
with open(os.path.join(od, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
