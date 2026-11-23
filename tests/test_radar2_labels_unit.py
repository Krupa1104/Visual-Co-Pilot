"""UNIT tests -- radar2 ocr_mode="labels" (23 Nov 2026). No model files needed.

Radar images are drawn with the radar team's own generator (radar2/src/generation/generate_radar.py),
so the label geometry is the real one. YOLO is replaced by the generator's ground-truth boxes and
PaddleOCR by a "perfect reader" that returns every ground-truth text line whose centre falls inside
the OCR crop -- including lines of overlapping neighbour labels. These tests therefore check the
crop geometry, label reading and label->aircraft matching logic, NOT OCR quality.
"""
import importlib.util
import os
import random
import sys

import numpy as np
import pytest

from conftest import REPO

RADAR_FILE = os.path.join(REPO, "radar2", "radar_system.py")
GEN_DIR = os.path.join(REPO, "radar2", "src", "generation")
if not os.path.exists(RADAR_FILE):
    pytest.skip(f"{RADAR_FILE} not found", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("radar2_radar_system_labels", RADAR_FILE)
rs = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(rs)
except ImportError as e:
    pytest.skip(f"radar_system imports failed: {e}", allow_module_level=True)
R = rs.RadarSystem

sys.path.insert(0, GEN_DIR)
gen = pytest.importorskip("generate_radar")


def make_scene(index, count, seed=7):
    rng = random.Random(seed * 1000 + index)
    image, contacts, meta = gen.generate_image(index, rng, count)
    return np.array(image)[:, :, ::-1].copy(), meta["contacts"]      # RGB -> BGR like cv2.imread


def system():
    r = object.__new__(R)                      # no models loaded
    r.ocr_mode = "labels"
    r.ocr_crop_pad_x, r.ocr_crop_pad_y = 4.0, 1.5
    return r


def perfect_reader(r, contacts):
    """Fake PaddleOCR: returns ground-truth lines inside the current label crop, in crop pixels."""
    lines = [tb for c in contacts for tb in c["text_boxes"]]
    state = {}
    real_rect = r._label_crop_rect

    def spy_rect(bbox, w, h):
        state["rect"] = real_rect(bbox, w, h)
        return state["rect"]
    r._label_crop_rect = spy_rect

    def run(crop):
        rx1, ry1, rx2, ry2 = state["rect"]
        s, pad = R.LABEL_OCR_UPSCALE, 16
        out = []
        for tb in lines:
            x1, y1, x2, y2 = tb["bbox_xyxy"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                X1, Y1 = (x1 - rx1) * s + pad, (y1 - ry1) * s + pad
                X2, Y2 = (x2 - rx1) * s + pad, (y2 - ry1) * s + pad
                out.append({"text": tb["text"], "confidence": 0.95,
                            "polygon": [[X1, Y1], [X2, Y1], [X2, Y2], [X1, Y2]],
                            "center": ((X1 + X2) / 2, (Y1 + Y2) / 2)})
        return out
    return run


def detections_for(contacts):
    return [{"bbox": tuple(c["aircraft_bbox_xyxy"]), "confidence": 0.9, "class_id": 0}
            for c in contacts]


def text_boxes_for(contacts):
    return [{"bbox": tuple(c["text_boxes"][0]["bbox_xyxy"]), "confidence": 0.9, "class_id": 1}
            for c in contacts]


def read_scene(image, contacts, drop_label=None):
    r = system()
    dets = detections_for(contacts)
    r._last_text_detections = [t for i, t in enumerate(text_boxes_for(contacts)) if i != drop_label]
    run = perfect_reader(r, contacts)
    items = r._ocr_label_blocks(image, run)
    if drop_label is not None:
        items += r._ocr_aircraft_crops(image, lambda crop: [], dets)   # fallback path runs, finds nothing
    return r._associate_ocr(dets, items, (image.shape[1], image.shape[0]))


# ----------------------------------------------------------------------------- single label
def test_read_label_block_by_keyword_and_by_position():
    cs = (100, 200, 166, 215)                  # callsign line box, text height 15
    def row(t, y):
        return {"text": t, "confidence": 0.9, "polygon": [[101, y], [180, y], [180, y + 15], [101, y + 15]]}
    f, used = R._read_label_block([row("SWR640", 200), row("ALT 31400", 225), row("HDG 253", 250)], cs)
    assert f == {"callsign": "SWR640", "altitude": 31400, "heading": 253} and len(used) == 3
    # OCR dropped the ALT / HDG keywords -> position decides (1st line below = ALT, 2nd = HDG)
    f, _ = R._read_label_block([row("SWR640", 200), row("A1T 31400", 225), row("H0G 253", 250)], cs)
    assert f == {"callsign": "SWR640", "altitude": 31400, "heading": 253}


def test_read_label_block_ignores_overlapping_neighbour_text():
    cs = (100, 200, 166, 215)
    rows = [{"text": "SWR640", "polygon": [[101, 200], [166, 200], [166, 215], [101, 215]]},
            {"text": "ALT 31400", "polygon": [[101, 225], [200, 225], [200, 240], [101, 240]]},
            {"text": "ALT 12000", "polygon": [[150, 228], [250, 228], [250, 243], [150, 243]]},  # neighbour
            {"text": "HDG 253", "polygon": [[101, 250], [180, 250], [180, 265], [101, 265]]}]
    f, _ = R._read_label_block(rows, cs)
    assert f == {"callsign": "SWR640", "altitude": 31400, "heading": 253}


# ----------------------------------------------------------------------------- whole scenes
def overprinted(contacts, i):
    """True if contact i's callsign line is drawn on top of another label's text (the glyphs overlap,
    so no real OCR can read it -- the perfect reader would still 'see' both)."""
    a = contacts[i]["text_boxes"][0]["bbox_xyxy"]
    return any(R._box_iou(a, tb["bbox_xyxy"]) > 0 for j, c in enumerate(contacts) if j != i
               for tb in c["text_boxes"])


@pytest.mark.parametrize("count", [1, 4, 7, 10, 12])
def test_every_label_read_and_matched_to_its_own_aircraft(count):
    for index in range(5):
        image, contacts = make_scene(index, count)
        records = read_scene(image, contacts)
        by_cs = {}
        for r in records:
            if r.get("callsign"):
                assert r["callsign"] not in by_cs, f"scene {index}: {r['callsign']} read twice"
                by_cs[r["callsign"]] = r
        for i, c in enumerate(contacts):
            if overprinted(contacts, i):
                continue
            rec = by_cs.get(c["callsign"])
            assert rec is not None, f"scene {index}: {c['callsign']} not read"
            assert rec["altitude"] == c["altitude"] and rec["heading"] == c["heading"], c["callsign"]
            assert rec["aircraft_index"] == i, f"scene {index}: {c['callsign']} matched to wrong aircraft"


def test_aircraft_without_label_box_keeps_others_correct():
    image, contacts = make_scene(0, 6)
    records = read_scene(image, contacts, drop_label=2)
    by_cs = {r["callsign"]: r for r in records if r.get("callsign")}
    assert contacts[2]["callsign"] not in by_cs
    for i, c in enumerate(contacts):
        if i != 2:
            assert by_cs[c["callsign"]]["aircraft_index"] == i


def test_unknown_ocr_mode_rejected():
    with pytest.raises(ValueError):
        R(base_dir=".", ocr_mode="nonsense", enable_ocr=False)
