"""INTEGRATION tests -- new radar module on real TEST images (marker `slow`).

Needs: radar2/models (YOLO + VQA checkpoints), PaddleOCR models (downloaded on first use),
and the radar team's split files in radar2/data/splits (test_frames.json, test.jsonl).
Questions come from the test split only (images neither the VQA model nor YOLO trained on), 5 per
question type below, fixed seed -> the same 15 questions every run. Uses the app's default OCR
setting (mobile models on crops around detected aircraft).

These are real checks: a wrong answer is a failing test, not a test to adjust.
"""
import json
import os
import random
import re
import statistics
import sys
import time

import pytest

from conftest import REPO

pytestmark = pytest.mark.slow
FINAL_RADAR = os.path.join(REPO, "radar2")
SPLITS = os.path.join(FINAL_RADAR, "data", "splits")
PER_TYPE = 5
LATENCY_TARGET_S = 10.0          # median per question, fixed 4 Nov 2026 (radar-eval decision rule)


def radar_test_sample():
    path = os.path.join(SPLITS, "test.jsonl")
    if not os.path.exists(path):
        return []
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rng = random.Random(0)
    out = []
    for qtype, keep in (("aircraft_count", lambda r: True),
                        ("quadrant", lambda r: True),
                        ("yes_no", lambda r: r["question"].lower().startswith("is any aircraft heading"))):
        pool = sorted([r for r in rows if r["question_type"] == qtype and keep(r)], key=lambda r: r["qa_id"])
        rng.shuffle(pool)
        out += pool[:PER_TYPE]
    return out


SAMPLE = radar_test_sample()


@pytest.fixture(scope="module")
def radar():
    if not os.path.isdir(os.path.join(FINAL_RADAR, "models")):
        pytest.skip(f"radar2 models not found in {FINAL_RADAR}")
    import importlib.util
    import torch  # noqa: F401  -- before paddle on Windows
    if FINAL_RADAR not in sys.path:
        sys.path.insert(0, FINAL_RADAR)          # radar2's own src/ and configs/ imports
    # Loaded under its own name: the old radar/radar_system.py may already be imported as "radar_system".
    mod = sys.modules.get("radar2_radar_system")
    if mod is None:
        spec = importlib.util.spec_from_file_location("radar2_radar_system",
                                                      os.path.join(FINAL_RADAR, "radar_system.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["radar2_radar_system"] = mod
    return mod.RadarSystem(base_dir=FINAL_RADAR)


TIMES = []


def correct(qtype, expected, answer):
    low = answer.lower()
    if qtype == "aircraft_count":
        m = re.search(r"\b(\d+)\b", answer)
        return bool(m) and (int(m.group(1)) >= 8 if expected == "8+" else int(m.group(1)) == int(expected))
    if qtype == "quadrant":
        return expected in low
    return low.startswith(expected)            # yes_no


@pytest.mark.parametrize("q", SAMPLE or [pytest.param(None, marks=pytest.mark.skip(
    reason=f"split files not found in {SPLITS}"))], ids=[q["qa_id"] for q in SAMPLE] or ["no-splits"])
def test_radar_answer_on_test_image(radar, q):
    t0 = time.perf_counter()
    out = radar.handle_question(os.path.join(FINAL_RADAR, q["image_path"]), q["question"])
    TIMES.append(time.perf_counter() - t0)
    assert correct(q["question_type"], q["answer"], out["answer"]), \
        f"{q['question']!r}: expected {q['answer']!r}, got {out['answer']!r} ({out.get('verdict')})"


def test_radar_median_latency():
    if not TIMES:
        pytest.skip("no radar questions were run")
    med = statistics.median(TIMES)
    print(f"radar median {med:.2f} s/question over {len(TIMES)} questions")
    assert med < LATENCY_TARGET_S
