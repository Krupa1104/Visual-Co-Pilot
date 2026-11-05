"""UNIT tests -- integrated.py callbacks (the GUI layer, radar card = radar2). No model files needed.

Checks what the user sees: the "Answered by" label for every radar answer route, which confidence
values are shown (YOLO average / OCR / VQA), and the cockpit "Routed by" chip of the rules-first
router. Fake radar/cockpit systems return fixed results.
"""
import importlib.util
import os
import re
import sys

import pytest

from conftest import REPO

pytest.importorskip("gradio")
APP_FILE = os.path.join(REPO, "integrated.py")
RADAR_FILE = os.path.join(REPO, "radar2", "radar_system.py")
if not os.path.exists(APP_FILE):
    pytest.skip(f"{APP_FILE} not found", allow_module_level=True)


@pytest.fixture(scope="module")
def app():
    argv = sys.argv
    sys.argv = ["integrated.py"]                  # the app parses args at import
    try:
        spec = importlib.util.spec_from_file_location("integrated", APP_FILE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                    # builds the UI, does not launch it
    finally:
        sys.argv = argv
    return mod


def use(app, monkeypatch, cockpit=None, radar=None):
    monkeypatch.setattr(app, "get_cockpit_system", lambda: (cockpit, None))
    monkeypatch.setattr(app, "get_radar_system", lambda: (radar, None))


class FakeRadar:
    def __init__(self, **result):
        self.result = {"kind": "vqa", "answer": "x", "image": None, "confidence": None,
                       "confidence_type": None, "ocr_confidence": None, "detections": [], **result}

    def handle_question(self, path, q):
        return self.result


# ----------------------------------------------------------------------------- radar
def test_every_radar_verdict_has_a_label(app):
    """Static check: each verdict string radar_system.py can return has an 'Answered by' label."""
    if not os.path.exists(RADAR_FILE):
        pytest.skip("radar2/radar_system.py not found")
    src = open(RADAR_FILE, encoding="utf-8").read()
    verdicts = set(re.findall(r'"verdict":\s*"([a-z_]+)"', src)) | set(re.findall(r'verdict = "([a-z_]+)"', src))
    assert verdicts, "no verdicts found -- has radar_system.py changed shape?"
    assert verdicts - set(app.RADAR_SOURCE_BY_VERDICT) == set()


@pytest.mark.parametrize("verdict, label", [
    ("yolo_geometry", "YOLO detector"),
    ("ocr_altitude_extreme", "OCR"),
    ("ocr_yolo_quadrant", "OCR"),
    ("vqa_answer", "VQA model"),
    ("out_of_scope", "out of scope"),
])
def test_radar_answered_by(app, monkeypatch, verdict, label):
    use(app, monkeypatch, radar=FakeRadar(verdict=verdict))
    _, meta, _, _ = app.radar_run_analysis("frame.png", "q")
    assert label in meta


def test_yolo_answer_shows_average_box_confidence(app, monkeypatch):
    boxes = [{"bbox": (0, 0, 1, 1), "confidence": 0.9}] * 3
    use(app, monkeypatch, radar=FakeRadar(verdict="yolo_geometry", confidence=0.95,
                                          confidence_type="detector", detections=boxes))
    _, meta, details, _ = app.radar_run_analysis("frame.png", "How many aircraft?")
    assert "avg of 3 boxes" in meta and "95.0%" in meta and "OCR confidence" not in meta


def test_ocr_answer_shows_ocr_confidence(app, monkeypatch):
    use(app, monkeypatch, radar=FakeRadar(verdict="yolo_ocr_match", confidence=0.97,
                                          confidence_type="detector", ocr_confidence=0.88))
    _, meta, _, _ = app.radar_run_analysis("frame.png", "What altitude is KAL511 at?")
    assert "OCR confidence" in meta and "88.0%" in meta


def test_out_of_scope_shows_no_confidence(app, monkeypatch):
    use(app, monkeypatch, radar=FakeRadar(verdict="out_of_scope", confidence=0.9, confidence_type="vqa"))
    _, meta, _, _ = app.radar_run_analysis("frame.png", "Tell me a joke")
    assert "confidence" not in meta.lower()


def test_radar_requires_image(app, monkeypatch):
    use(app, monkeypatch, radar=object())
    assert "upload" in app.radar_run_analysis(None, "How many aircraft?")[0].lower()


def test_radar_does_not_get_the_cockpit_llm():
    """Radar's relevance check must not depend on which card was opened first (4 Nov fix)."""
    src = open(APP_FILE, encoding="utf-8").read()
    body = src[src.index("def _get_radar_system"):src.index("def _shared_llm_for_radar")]
    assert re.search(r"^\s*shared = \{\}", body, re.MULTILINE)


# ----------------------------------------------------------------------------- cockpit
class FakeCockpit:
    model_info = "fake"
    llm_mode = "generate"

    def __init__(self, router):
        self.router = router

    def answer(self, image, question):
        return {"params": dict(altitude=5518, airspeed=110, heading=18, pitch=1.3, roll=11.2, vs=60,
                               turn=0.62, rpm=2410),
                "phase": "level_flight", "phase_confidence": 1.0, "intent": "get_heading", "confidence": 1.0,
                "answer": "The heading indicator shows about 018°.", "router": self.router, "panel": None,
                "crop_mode": "already_panel", "panel_similarity": 0.9, "warnings": [],
                "timings": {"preprocess_ms": 20, "inference_ms": 200, "routing_ms": 0.1, "total_ms": 220}}


def test_cockpit_requires_image(app, monkeypatch):
    use(app, monkeypatch, cockpit=FakeCockpit("rules"))
    assert "upload" in app.cockpit_run_analysis(None, "What heading?")[0].lower()


def test_cockpit_requires_question(app, monkeypatch):
    use(app, monkeypatch, cockpit=FakeCockpit("rules"))
    assert "question" in app.cockpit_run_analysis("img", "   ")[0].lower()


def test_cockpit_answer_and_highlighted_tile(app, monkeypatch):
    use(app, monkeypatch, cockpit=FakeCockpit("rules"))
    answer, meta, readings, details, _ = app.cockpit_run_analysis("img", "What heading are we on?")
    assert "018" in answer
    assert readings.count("vc-tile active") == 1 and "Heading" in readings


def test_cockpit_out_of_scope_shows_no_readings(app, monkeypatch):
    fake = FakeCockpit("rules")
    result = fake.answer(None, "")
    result.update(intent=None, confidence=0.0)
    fake.answer = lambda image, question: result
    use(app, monkeypatch, cockpit=fake)
    answer, _, readings, _, _ = app.cockpit_run_analysis("img", "Tell me a joke")
    assert answer == app.COCKPIT_OUT_OF_SCOPE
    assert "vc-tile" not in readings


@pytest.mark.parametrize("router, label", [("rules", "rules"), ("qwen", "Qwen")])
def test_cockpit_shows_which_router_answered(app, monkeypatch, router, label):
    use(app, monkeypatch, cockpit=FakeCockpit(router))
    _, meta, _, _, _ = app.cockpit_run_analysis("img", "What heading are we on?")
    assert "Routed by" in meta and label in meta
