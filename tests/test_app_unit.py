"""UNIT tests -- integrated.py callbacks (the GUI layer). No model files needed.

The cockpit/radar systems are replaced by fakes, so these check what the GUI does with
a result: input validation, out-of-scope handling, which reading tile is highlighted,
and that radar answers are labelled with their true source (CNN / YOLO / metadata).
"""
import importlib.util
import os
import sys

import pytest

from conftest import REPO

pytest.importorskip("gradio")


@pytest.fixture(scope="module")
def app():
    argv = sys.argv
    sys.argv = ["integrated.py"]                      # integrated.py parses args at import
    try:
        spec = importlib.util.spec_from_file_location("integrated", os.path.join(REPO, "integrated.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                  # builds the UI, does not launch it
    finally:
        sys.argv = argv
    return mod


def cockpit_result(intent="get_heading", conf=1.0):
    return {"params": dict(altitude=5518, airspeed=110, heading=18, pitch=1.3, roll=11.2, vs=60,
                           turn=0.62, rpm=2410),
            "phase": "level_flight", "phase_confidence": 1.0, "intent": intent, "confidence": conf,
            "answer": "The heading indicator shows about 018°.", "panel": None,
            "crop_mode": "already_panel", "panel_similarity": 0.9, "warnings": [],
            "timings": {"preprocess_ms": 20, "inference_ms": 200, "routing_ms": 9000, "total_ms": 9220}}


class FakeCockpit:
    model_info = "fake"

    def __init__(self, result):
        self.result = result

    def answer(self, image, question):
        return self.result


def use(app, monkeypatch, cockpit=None, radar=None):
    monkeypatch.setattr(app, "get_cockpit_system", lambda: (cockpit, None))
    monkeypatch.setattr(app, "get_radar_system", lambda: (radar, None))


def test_cockpit_requires_image(app, monkeypatch):
    use(app, monkeypatch, cockpit=FakeCockpit(cockpit_result()))
    assert "upload" in app.cockpit_run_analysis(None, "What heading?")[0].lower()


def test_cockpit_requires_question(app, monkeypatch):
    use(app, monkeypatch, cockpit=FakeCockpit(cockpit_result()))
    assert "question" in app.cockpit_run_analysis("img", "   ")[0].lower()


def test_cockpit_answer_and_highlighted_tile(app, monkeypatch):
    use(app, monkeypatch, cockpit=FakeCockpit(cockpit_result("get_heading")))
    answer, meta, readings, details, _ = app.cockpit_run_analysis("img", "What heading are we on?")
    assert "018" in answer
    assert readings.count("vc-tile active") == 1 and "Heading" in readings
    assert "Routing" in meta


def test_cockpit_out_of_scope_shows_no_readings(app, monkeypatch):
    use(app, monkeypatch, cockpit=FakeCockpit(cockpit_result(intent=None, conf=0.0)))
    answer, _, readings, _, _ = app.cockpit_run_analysis("img", "Tell me a joke")
    assert answer == app.COCKPIT_OUT_OF_SCOPE
    assert "vc-tile" not in readings


@pytest.mark.parametrize("verdict, kind, label", [
    ("yolo_first", "vqa", "YOLO"),
    ("metadata_altitude_extreme", "vqa", "metadata"),
    ("confident", "vqa", "radar CNN"),
    (None, "irrelevant", "out of scope"),
])
def test_radar_answer_source_label(app, monkeypatch, verdict, kind, label):
    class FakeRadar:
        def handle_question(self, path, q):
            return {"kind": kind, "answer": "x", "image": None, "confidence": None, "verdict": verdict}
    use(app, monkeypatch, radar=FakeRadar())
    _, meta, _, _ = app.radar_run_analysis("frame.png", "q")
    assert label in meta


def test_radar_requires_image(app, monkeypatch):
    use(app, monkeypatch, radar=object())
    assert "upload" in app.radar_run_analysis(None, "How many aircraft?")[0].lower()
