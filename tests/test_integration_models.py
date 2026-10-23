"""INTEGRATION tests -- the real trained models, end to end. Marked `slow`.

Cockpit: CopilotSystemV2 (crop -> CockpitNetV2 -> decode -> router -> answer) on hold-out
images (captured after the final model was frozen), 5 per phase. Each image is one test:
it passes only if ALL 8 gauges are within the aviation tolerances AND the phase is right.
Radar:   RadarSystem on frames from rendered_frames.json: the "how many aircraft" answer is
compared with the true number of aircraft in that frame. Each frame is one test.

The TF-IDF router is used (use_llm=False) so these run without downloading Qwen;
the Qwen router has its own tests in test_llm_router.py (marker `llm`).
"""
import json
import os
import random
import re

import pytest
from PIL import Image

from conftest import HOLDOUT, HOLDOUT_DIR, RADAR_BASE, TOL, YOLO_ROOT

pytestmark = pytest.mark.slow
GAUGES = ["pitch", "roll", "airspeed", "altitude", "heading", "vs", "turn", "rpm"]


# ----------------------------------------------------------------------------- cockpit
def gauge_error(k, pred, true):
    if k == "heading":
        d = abs(pred - true) % 360
        return min(d, 360 - d)
    return abs(pred - true)


@pytest.mark.parametrize("row", HOLDOUT or [pytest.param(None, marks=pytest.mark.skip(
    reason=f"hold-out labels.csv not found in {HOLDOUT_DIR}"))],
    ids=[r["image"] for r in HOLDOUT] or ["no-holdout"])
def test_cockpit_holdout_image(cockpit, row):
    img = Image.open(os.path.join(HOLDOUT_DIR, row["phase"], row["image"]))
    pred, info = cockpit.run_perception(img)
    errors = {k: round(gauge_error(k, pred[k], row[k]), 2) for k in GAUGES}
    wrong = {k: e for k, e in errors.items() if e > TOL[k]}
    assert not wrong, f"outside tolerance: {wrong}"
    assert pred["phase"] == row["phase"]
    assert not info["warnings"], info["warnings"]


@pytest.mark.parametrize("question, intent, gauge", [
    ("What is the current altitude?", "get_altitude", "altitude"),
    ("How fast are we going?", "get_airspeed", "airspeed"),
    ("What is the engine RPM?", "get_rpm", "rpm"),
])
def test_cockpit_question_to_answer(cockpit, question, intent, gauge):
    """Full path: image + question -> routed intent -> answer text containing the reading."""
    if not HOLDOUT:
        pytest.skip("no hold-out images")
    row = next(r for r in HOLDOUT if r["phase"] == "level_flight")
    out = cockpit.answer(Image.open(os.path.join(HOLDOUT_DIR, row["phase"], row["image"])), question)
    assert out["intent"] == intent
    number = float(re.search(r"(\d[\d,]*)", out["answer"].split("about")[-1]).group(1).replace(",", ""))
    assert abs(number - row[gauge]) <= TOL[gauge] + 1     # +1: answer text is rounded


def test_cockpit_full_screenshot_is_cropped(cockpit):
    """A full 16:10 screenshot must be auto-cropped (simulated by pasting a hold-out crop
    into a black 2880x1800 frame at the capture crop position)."""
    if not HOLDOUT:
        pytest.skip("no hold-out images")
    row = HOLDOUT[0]
    crop = Image.open(os.path.join(HOLDOUT_DIR, row["phase"], row["image"])).convert("RGB")
    screen = Image.new("RGB", (2880, 1800))
    screen.paste(crop, (round(0.365 * 2880), round(0.465 * 1800)))
    pred, info = cockpit.run_perception(screen)
    assert info["crop_mode"] == "cropped_screenshot"
    assert abs(pred["altitude"] - row["altitude"]) <= TOL["altitude"]


# ----------------------------------------------------------------------------- radar
def radar_frames(n=20, seed=0):
    path = os.path.join(RADAR_BASE, "data", "raw", "rendered_frames.json")
    img_dir = os.path.join(RADAR_BASE, "data", "raw", "images")
    if not os.path.exists(path):
        return []
    frames = [f for f in json.load(open(path))
              if os.path.exists(os.path.join(img_dir, f"{f['frame_id']}.png"))]
    random.Random(seed).shuffle(frames)
    return [{"image": os.path.join(img_dir, f"{f['frame_id']}.png"), "n": len(f["aircraft"]),
             "id": str(f["frame_id"])} for f in frames[:n]]


RADAR = radar_frames()


@pytest.fixture(scope="module")
def radar():
    if not os.path.isdir(RADAR_BASE):
        pytest.skip(f"radar_base not found at {RADAR_BASE}")
    from radar_system import RadarSystem
    return RadarSystem(base_dir=RADAR_BASE, yolo_root=YOLO_ROOT)


@pytest.mark.parametrize("frame", RADAR or [pytest.param(None, marks=pytest.mark.skip(
    reason=f"radar frames not found in {RADAR_BASE}"))], ids=[f["id"] for f in RADAR] or ["no-radar"])
def test_radar_aircraft_count(radar, frame):
    out = radar.handle_question(frame["image"], "How many aircraft are visible on the radar?")
    answer = out["answer"]
    assert "specify" not in answer.lower(), answer
    counted = int(re.search(r"(\d+) aircraft", answer).group(1))
    assert counted == frame["n"], f"answered {counted}, frame has {frame['n']}: {answer}"
