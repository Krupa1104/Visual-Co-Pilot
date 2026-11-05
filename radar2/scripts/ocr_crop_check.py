r"""
ocr_crop_check.py -- second OCR speed check: mobile OCR models on (a) the full image versus
(b) only the area around each aircraft that YOLO found (where its label box is drawn).

Run 1 (ocr_speed_check.py, 4 Nov, 10 images): current v6 medium models 28.9 s/image, v5 mobile models
8.1 s/image, both read 19/20 ground-truth callsigns; mkldnn fails with the same error as before.

For the same kind of sample this script reports, per strategy: OCR seconds per image (YOLO time
reported separately), ground-truth callsigns read, and how many text items OCR returned.
It only measures; radar_system.py is not changed.

Run from the repo root (venv_final), plugged in:
  python radar2\scripts\ocr_crop_check.py --n 10
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
import time

import torch  # noqa: F401  -- before paddle on Windows
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
from radar_system import RadarSystem  # noqa: E402

CALLSIGN = re.compile(r"\b[A-Z]{2,5}\d{1,5}[A-Z]?\b")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pad_x", type=float, default=4.0, help="crop extends this many box widths left/right")
ap.add_argument("--pad_y", type=float, default=1.5, help="crop extends this many box heights up/down")
args = ap.parse_args()

gt = {}
with open(os.path.join(BASE, "data", "qa_complete", "annotations.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["question_type"] == "callsign":
            gt.setdefault(r["image_path"], set()).add(r["answer"].upper())
paths = sorted(gt)
random.Random(args.seed).shuffle(paths)
paths = paths[:args.n + 1]

print("Loading YOLO + VQA model (OCR off inside RadarSystem; this script makes its own OCR) ...")
rs = RadarSystem(base_dir=BASE, enable_ocr=False)

from paddleocr import PaddleOCR  # noqa: E402
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                enable_mkldnn=False, text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec")


def ocr_rows(img):
    rows = []
    for res in ocr.predict(img) or []:
        rows += RadarSystem._extract_paddle_result(res)
    return rows


def crops_around(img, boxes):
    h, w = img.shape[:2]
    out = []
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        cx1, cx2 = max(0, int(x1 - args.pad_x * bw)), min(w, int(x2 + args.pad_x * bw))
        cy1, cy2 = max(0, int(y1 - args.pad_y * bh)), min(h, int(y2 + args.pad_y * bh))
        if cx2 > cx1 and cy2 > cy1:
            out.append(img[cy1:cy2, cx1:cx2])
    return out


results = {"full image": [], "crops around YOLO aircraft": []}
yolo_times = []
for i, p in enumerate(paths):
    full = os.path.join(BASE, p)
    img = cv2.imread(full)
    t0 = time.perf_counter()
    boxes = [d["bbox"] for d in rs._detect_aircraft(full)]
    yolo_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rows_full = ocr_rows(img)
    full_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rows_crop = []
    for c in crops_around(img, boxes):
        rows_crop += ocr_rows(c)
    crop_s = time.perf_counter() - t0

    if i == 0:                         # warm-up image, not counted
        continue
    yolo_times.append(yolo_s)
    for name, rows, secs in (("full image", rows_full, full_s), ("crops around YOLO aircraft", rows_crop, crop_s)):
        found = set()
        for r in rows:
            found |= set(CALLSIGN.findall(r["text"].upper()))
        results[name].append((secs, len(gt[p] & found), len(gt[p]), len(rows), len(boxes)))

print(f"\n=== {args.n} images (+1 warm-up), seed {args.seed}, mobile OCR models")
print(f"YOLO detection: median {statistics.median(yolo_times):.2f} s/image (needed by both strategies "
      f"in the app anyway)")
for name, rs_ in results.items():
    secs = [r[0] for r in rs_]
    hit, tot = sum(r[1] for r in rs_), sum(r[2] for r in rs_)
    items = statistics.median([r[3] for r in rs_])
    print(f"{name}: OCR median {statistics.median(secs):.2f} s/image (min {min(secs):.2f}, max {max(secs):.2f}); "
          f"ground-truth callsigns read {hit}/{tot}; median text items {items:.0f}")
