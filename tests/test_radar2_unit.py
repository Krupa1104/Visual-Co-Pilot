"""UNIT tests -- new radar module (radar2/radar_system.py). No model files needed.

Covers the 4-5 Nov 2026 changes: OCR line joining, callsign look-alike digit fix, ALT/HDG parsing,
heading -> 8-point compass, quadrants around the radar SCOPE centre (not the image centre), the
heading yes/no and quadrant question routes, and the OCR settings check. YOLO and OCR are replaced
by fixed fake results, so every expected answer is known exactly.
"""
import importlib.util
import os

import pytest

from conftest import REPO

FINAL_RADAR = os.path.join(REPO, "radar2")
RADAR_FILE = os.path.join(FINAL_RADAR, "radar_system.py")
if not os.path.exists(RADAR_FILE):
    pytest.skip(f"{RADAR_FILE} not found", allow_module_level=True)

# Loaded under its own name so it cannot clash with the old radar/radar_system.py.
_spec = importlib.util.spec_from_file_location("radar2_radar_system", RADAR_FILE)
rs = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(rs)
except ImportError as e:                       # torch / cv2 / torchvision missing
    pytest.skip(f"radar_system imports failed: {e}", allow_module_level=True)
R = rs.RadarSystem

W = H = 1280                                   # dataset image size
SCOPE_Y = 655                                  # measured scope centre y (image centre is 640)


def row(text, x1, y1, x2, y2, conf=0.99):
    return {"text": text, "confidence": conf, "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "center": ((x1 + x2) / 2, (y1 + y2) / 2)}


# ----------------------------------------------------------------------------- OCR text handling
def test_line_joining_rebuilds_alt_and_hdg_lines():
    # what the mobile OCR models return for one label box (ocr_text_dump.py, 4 Nov)
    rows = [row("JAL508", 20, 10, 90, 28), row("ALT", 20, 35, 55, 53), row("30400", 62, 35, 120, 53),
            row("HDG", 20, 60, 55, 78), row("353", 62, 60, 95, 78)]
    texts = sorted(r["text"] for r in R._merge_text_lines(rows))
    assert texts == ["ALT 30400", "HDG 353", "JAL508"]


def test_line_joining_keeps_two_labels_side_by_side_apart():
    rows = [row("ALT", 20, 35, 55, 53), row("30400", 62, 35, 120, 53),
            row("ALT", 300, 36, 335, 54), row("12000", 342, 36, 400, 54)]
    assert sorted(r["text"] for r in R._merge_text_lines(rows)) == ["ALT 12000", "ALT 30400"]


@pytest.mark.parametrize("text, field, value", [
    ("ALT 30400", "altitude", 30400),
    ("HDG 353", "heading", 353),
    ("HDG 077", "heading", 77),
    ("QTR811", "callsign", "QTR811"),
    ("KLM315", "callsign", "KLM315"),
    ("IG0452", "callsign", "IGO452"),          # OCR read the letter O as zero
    ("IGO452", "callsign", "IGO452"),
])
def test_parse_label_fields(text, field, value):
    assert R._parse_visible_label(text)[field] == value


@pytest.mark.parametrize("text", ["RDR-942", "SYN-02966", "TRACKS: 03", "RANGE: 50 NM"])
def test_screen_text_is_not_a_callsign(text):
    assert R._parse_visible_label(text)["callsign"] is None


# ----------------------------------------------------------------------------- compass / quadrants
@pytest.mark.parametrize("deg, compass", [(0, "N"), (22, "N"), (23, "NE"), (96, "E"), (173, "S"),
                                          (248, "W"), (337, "NW"), (338, "N"), (360, "N")])
def test_heading_to_8_point_compass(deg, compass):
    assert rs.heading_to_compass8(deg) == compass


def test_quadrant_uses_scope_centre_not_image_centre():
    # box centre y = 650: below the image centre (640) but ABOVE the scope centre (655)
    box = (600, 645, 620, 655)
    assert R._quadrant_for_box(box, W, H) == "top-left"
    assert R._compass_quadrant("top-left") == "north-west"


@pytest.mark.parametrize("screen, compass", [("top-left", "north-west"), ("top-right", "north-east"),
                                             ("bottom-left", "south-west"), ("bottom-right", "south-east")])
def test_screen_to_compass_quadrant(screen, compass):
    assert R._compass_quadrant(screen) == compass


@pytest.mark.parametrize("question, direction", [
    ("Is any aircraft heading E?", "E"), ("Is any aircraft heading NW?", "NW"),
    ("Are any planes heading north-east?", "north-east"),
])
def test_heading_yes_no_question_is_recognised(question, direction):
    assert rs.HEADING_YESNO.search(question).group("dir") == direction


@pytest.mark.parametrize("question", ["What compass direction is aircraft KAL511 heading?",
                                      "How many aircraft are present in the radar image?"])
def test_other_questions_are_not_heading_yes_no(question):
    assert rs.HEADING_YESNO.search(question) is None


# ----------------------------------------------------------------------------- structured routes
# 3 fake aircraft (YOLO boxes) with their labels drawn to the left, like the real images:
#   KAL511 top-right, heading 095 (E);  DAL901 bottom-left, heading 200 (S);  AFR302 bottom-right, 330 (NW)
AIRCRAFT = [("KAL511", (900, 300, 936, 340), "095"), ("DAL901", (300, 900, 336, 940), "200"),
            ("AFR302", (900, 900, 936, 940), "330")]


@pytest.fixture
def image(tmp_path):
    from PIL import Image
    p = tmp_path / "radar.png"
    Image.new("RGB", (W, H)).save(p)
    return str(p)


@pytest.fixture
def radar():
    r = object.__new__(R)                      # routing logic only -- no models loaded
    r._last_ocr_confidence = 0.99
    items = []
    for callsign, (x1, y1, x2, y2), hdg in AIRCRAFT:
        lx = x1 - 130                          # label box left of the aircraft
        items += [row(callsign, lx, y1 - 10, lx + 70, y1 + 8), row(f"ALT 30400", lx, y1 + 15, lx + 100, y1 + 33),
                  row(f"HDG {hdg}", lx, y1 + 40, lx + 80, y1 + 58)]
    r._ocr_text = lambda path, detections=None: items
    return r


def dets(boxes):
    return [{"bbox": b, "confidence": 0.98, "class_id": 0, "name": "aircraft"} for b in boxes]


DETS = dets([b for _, b, _ in AIRCRAFT])


def test_heading_yes(radar, image):
    out = radar._answer_structured(image, "Is any aircraft heading E?", DETS)
    assert out["verdict"] == "ocr_heading_yesno" and out["answer"].startswith("Yes") and "KAL511" in out["answer"]


def test_heading_no(radar, image):
    out = radar._answer_structured(image, "Is any aircraft heading W?", DETS)
    assert out["answer"].startswith("No")


def test_heading_question_with_no_aircraft_is_no(radar, image):
    out = radar._answer_structured(image, "Is any aircraft heading S?", [])
    assert out["verdict"] == "yolo_no_aircraft" and out["answer"].startswith("No")


def test_quadrant_of_callsign(radar, image):
    out = radar._answer_structured(image, "Which quadrant contains aircraft DAL901?", DETS)
    assert out["verdict"] == "ocr_yolo_quadrant" and "south-west" in out["answer"]


def test_quadrant_of_unknown_callsign(radar, image):
    out = radar._answer_structured(image, "Which quadrant contains aircraft QTR811?", DETS)
    assert out["verdict"] == "yolo_ocr_match_not_found"


@pytest.mark.parametrize("quadrant, expect", [("south-east", "Yes"), ("north-west", "No")])
def test_any_aircraft_in_compass_quadrant(radar, image, quadrant, expect):
    out = radar._answer_structured(image, f"Is there any aircraft in the {quadrant} quadrant?", DETS)
    assert out["verdict"] == "yolo_compass_quadrant" and out["answer"].startswith(expect)


def test_count_question_is_left_to_yolo_route(radar, image):
    assert radar._answer_structured(image, "How many aircraft are present in the radar image?", DETS) is None


# ----------------------------------------------------------------------------- settings
def test_invalid_ocr_setting_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        R(base_dir=str(tmp_path), ocr_models="large")
