"""UNIT tests -- radar question routing (radar/radar_system.py). No model files needed.

YOLO is replaced by 5 fake detections at known positions, so every expected answer is
known exactly:
    top-left: 2   top-right: 1   bottom-left: 0   bottom-right: 2      (total 5)
    left side (x <= 40%): 1   right side (x >= 60%): 3
The CNN is replaced by a stub that answers "CNN" -- a test can then see whether a
question was (correctly) sent to the CNN instead of being answered from YOLO boxes.
Each test is one requirement from the sample questions / test set question types.
"""
import pytest

rs = pytest.importorskip("radar_system", reason="radar/radar_system.py or its imports (torch, ultralytics) missing")

# (x, y) box centres in the 224x224 image (y grows downwards); altitudes for the metadata route.
BOXES = [(40, 50, 31000), (90, 30, 12000), (180, 60, 25000), (150, 170, 9000), (200, 200, 40000)]


@pytest.fixture
def radar():
    r = object.__new__(rs.RadarSystem)              # routing logic only -- nothing loaded
    fake = [{"cx": x, "cy": y, "xmin": x - 4, "ymin": y - 4, "xmax": x + 4, "ymax": y + 4,
             "confidence": 0.9, "record": {"callsign": f"AB{i}", "altitude": alt, "heading": 90,
                        "nx": x / 224, "ny": 1 - y / 224}}
            for i, (x, y, alt) in enumerate(BOXES)]
    r.callsign_index = {f"AB{i}": [fake[i]["record"]] for i in range(len(fake))}
    r.classify_relevance = lambda q: "weather" not in q.lower()
    r._yolo_aircraft_records = lambda path, **kw: fake
    r._yolo_count = lambda path, conf=0.25: len(fake)
    r.locate_and_highlight = lambda q: None
    r.evaluate_and_answer = lambda path, q: {"answer": "CNN", "raw_answer": "CNN",
                                             "confidence": 0.9, "verdict": "confident"}
    return r


def ask(radar, q):
    return radar.handle_question("frame_0001.png", q)


def test_total_count(radar):
    out = ask(radar, "How many aircraft are visible on the radar?")
    assert " 5 aircraft" in out["answer"] and "specify" not in out["answer"].lower()


@pytest.mark.parametrize("q, n, quadrant", [
    ("How many aircraft are in the upper-left quadrant?", 2, "top-left"),
    ("How many aircraft are in the top right quadrant?", 1, "top-right"),
    ("How many aircraft are in the bottom-left quadrant?", 0, "bottom-left"),
    ("How many aircraft are in the lower right quadrant?", 2, "bottom-right"),
])
def test_count_in_one_quadrant(radar, q, n, quadrant):
    a = ask(radar, q)["answer"]
    assert f" {n} aircraft" in a and quadrant in a


def test_quadrant_with_most_aircraft_reports_tie(radar):
    a = ask(radar, "Which quadrant of the radar has the most aircraft?")["answer"].lower()
    assert "top-left" in a and "bottom-right" in a          # 2 vs 2 -> tie


@pytest.mark.parametrize("q, expect", [
    ("Is there any aircraft on the left side of the radar?", "yes"),
    ("Is there any aircraft on the right side of the radar?", "yes"),
    ("Is there any aircraft near the center of the radar?", "no"),
])
def test_is_there_aircraft_in_region(radar, q, expect):
    a = ask(radar, q)["answer"].lower()
    assert a.startswith(expect) and "specify" not in a


@pytest.mark.parametrize("q, expect", [("Are there more than 5 aircraft on the radar?", "no"),
                                       ("Are there more than 3 aircraft on the radar?", "yes")])
def test_more_than_n(radar, q, expect):
    assert ask(radar, q)["answer"].lower().startswith(expect)


def test_altitude_threshold_count_goes_to_cnn_not_total(radar):
    """YOLO can't read altitudes: 'how many above 30,000 ft' must not be answered with the total."""
    out = ask(radar, "How many aircraft are flying above 30,000 feet?")
    assert out["answer"] == "CNN"


def test_highest_altitude_callsign(radar):
    assert "AB4" in ask(radar, "What is the callsign of the aircraft with the highest altitude?")["answer"]


def test_off_topic_question_rejected(radar):
    assert ask(radar, "What's the weather like today?")["kind"] == "irrelevant"