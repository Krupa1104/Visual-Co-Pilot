r"""
ocr_speed_check.py -- how long does PaddleOCR take per radar image, and which settings are faster
without losing callsigns?

For N radar images it runs full-image OCR with several PaddleOCR settings and reports, per setting:
  * seconds per image (after one warm-up image),
  * how many of the image's ground-truth callsigns were read (from data/qa_complete/annotations.jsonl,
    callsign questions; only the aircraft those questions ask about, so this is a lower bound),
  * the error, if a setting does not work on this PC (e.g. the mkldnn error).
It also says how long the CURRENT app setting takes per image, so we can tell whether the ~60 s per
question comes from full-image OCR or from the extra crop passes in radar_system._ocr_text.
It only measures; radar_system.py is not changed.

Run from the repo root (venv_final):
  python radar2\scripts\ocr_speed_check.py --n 10
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
import time

import torch  # noqa: F401  -- must be imported before paddle on Windows (shm.dll / WinError 127)
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                       # radar2/
sys.path.insert(0, BASE)
from radar_system import RadarSystem  # noqa: E402  (reuses its PaddleOCR result parser)

CALLSIGN = re.compile(r"\b[A-Z]{2,5}\d{1,5}[A-Z]?\b")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10, help="number of radar images")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

# ground truth: callsign answers per image
gt = {}
with open(os.path.join(BASE, "data", "qa_complete", "annotations.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["question_type"] == "callsign":
            gt.setdefault(r["image_path"], set()).add(r["answer"].upper())
paths = sorted(gt)
random.Random(args.seed).shuffle(paths)
paths = paths[:args.n + 1]                          # +1 for the warm-up image

COMMON = dict(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
              use_textline_orientation=False)
CONFIGS = [
    ("current app setting (default models, mkldnn off)", dict(COMMON, enable_mkldnn=False)),
    ("detection on image resized to max side 960", dict(COMMON, enable_mkldnn=False,
                                                         text_det_limit_side_len=960,
                                                         text_det_limit_type="max")),
    ("mobile det + rec models (PP-OCRv5_mobile)", dict(
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
        enable_mkldnn=False, text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec")),
    ("mkldnn ON (was off because of an error)", dict(COMMON, enable_mkldnn=True)),
]


def read_callsigns(ocr, img):
    found = set()
    for res in ocr.predict(img) or []:
        for row in RadarSystem._extract_paddle_result(res):
            found |= set(CALLSIGN.findall(row["text"].upper()))
    return found


from paddleocr import PaddleOCR  # noqa: E402

print(f"{args.n} images (+1 warm-up), seed {args.seed}\n")
summary = []
for name, kw in CONFIGS:
    print(f"--- {name}")
    try:
        ocr = PaddleOCR(**kw)
        read_callsigns(ocr, cv2.imread(os.path.join(BASE, paths[0])))      # warm-up, not timed
        times, hit, total = [], 0, 0
        for p in paths[1:]:
            img = cv2.imread(os.path.join(BASE, p))
            t0 = time.perf_counter()
            found = read_callsigns(ocr, img)
            times.append(time.perf_counter() - t0)
            hit += len(gt[p] & found)
            total += len(gt[p])
        line = (f"median {statistics.median(times):.2f} s/image (min {min(times):.2f}, "
                f"max {max(times):.2f}); ground-truth callsigns read: {hit}/{total}")
    except Exception as e:                           # setting not supported / mkldnn error etc.
        line = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
    print("   " + line + "\n", flush=True)
    summary.append((name, line))

print("=== summary")
for name, line in summary:
    print(f"{name}: {line}")
