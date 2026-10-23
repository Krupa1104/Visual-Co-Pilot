"""UNIT tests -- cockpit module (cockpit2/gui). No trained model files needed.

Covers: altimeter / needle-angle decoding, heading wrap-around, model output shapes,
auto-crop, panel-similarity check, answer templates and the question router fallbacks.
"""
import math

import pytest
import torch
from PIL import Image

from conftest import REFERENCE_PANEL, need
import cockpit_model_v2 as cm
import copilot_core_v2 as core


# ----------------------------------------------------------------------------- decoding
def angle(value, period):
    """(sin, cos) of a needle at `value` on a dial with `period` units per turn."""
    a = 2 * math.pi * value / period
    return math.sin(a), math.cos(a)


@pytest.mark.parametrize("true_alt, coarse_err", [
    (5300, -200), (5300, +450), (999, 300), (1001, -300), (9480, 420), (16, 40), (7050, -490)])
def test_altimeter_long_hand_recovers_exact_altitude(true_alt, coarse_err):
    """Long hand (1,000 ft/turn) gives the exact hundreds as long as the coarse
    regression is within +-500 ft -- the reason the needle-angle head was added."""
    s, c = angle(true_alt, cm.ALT_PERIOD)
    assert cm.combine_altitude(true_alt + coarse_err, s, c, cm.ALT_PERIOD) == pytest.approx(true_alt, abs=1e-6)


def test_altimeter_cascade_short_then_long_hand():
    """Coarse error of 2,400 ft (wrong thousand) is fixed by the short hand, then the long hand."""
    true_alt, coarse = 6730.0, 4330.0
    est = cm.combine_altitude(coarse, *angle(true_alt, cm.ALT_SHORT_PERIOD), cm.ALT_SHORT_PERIOD)
    est = cm.combine_altitude(est, *angle(true_alt, cm.ALT_PERIOD), cm.ALT_PERIOD)
    assert est == pytest.approx(true_alt, abs=1e-6)


def _fake_output(norm, alt_true=6730.0, heading_deg=350.0, rpm_true=2410.0, airspeed_true=0.0,
                 phase_idx=3):
    """Model output where the plain regression says 'mean' for every gauge, and the angle
    heads carry the true values -- decode() must return the true values."""
    out = {"reg": torch.zeros(1, len(cm.SCALARS)),
           "heading": torch.tensor([[math.sin(math.radians(heading_deg)), math.cos(math.radians(heading_deg))]]),
           "phase": torch.full((1, len(cm.PHASES)), -5.0),
           "alt_short": torch.tensor([angle(alt_true, cm.ALT_SHORT_PERIOD)]),
           "alt_fine": torch.tensor([angle(alt_true, cm.ALT_PERIOD)]),
           "needle_rpm": torch.tensor([angle(rpm_true, cm.NEEDLE_PERIODS["rpm"])]),
           "needle_airspeed": torch.tensor([angle(airspeed_true, cm.NEEDLE_PERIODS["airspeed"])])}
    out["phase"][0, phase_idx] = 5.0
    return out


NORM = {k: {"mean": m, "std": 1.0} for k, m in
        {"pitch": 2.0, "roll": 0.0, "airspeed": 3.0, "altitude": 6000.0, "vs": 0.0,
         "turn": 0.0, "rpm": 2300.0}.items()}


def test_decode_uses_angle_heads_and_wraps_heading():
    row = cm.decode(_fake_output(NORM), NORM)[0]
    assert row["altitude"] == pytest.approx(6730.0, abs=1e-3)
    assert row["rpm"] == pytest.approx(2410.0, abs=1e-3)
    assert row["heading"] == pytest.approx(350.0, abs=1e-3)       # not -10
    assert row["phase"] == "climb"
    assert 0.99 < row["phase_confidence"] <= 1.0


def test_decode_airspeed_never_negative():
    """A needle just below 0 (angle of 249 kt on a 250 kt turn) must decode to 0, not -1 kt."""
    out = _fake_output(NORM, airspeed_true=249.0)                  # coarse = 3 kt (the mean)
    assert cm.decode(out, NORM)[0]["airspeed"] == 0.0


def test_model_output_shapes():
    model = cm.CockpitNetV2(pretrained=False, needles=["rpm", "airspeed"]).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, cm.IMG_H, cm.IMG_W))
    assert out["reg"].shape == (2, len(cm.SCALARS))
    assert out["phase"].shape == (2, len(cm.PHASES))
    for k in ("heading", "alt_fine", "alt_short", "needle_rpm", "needle_airspeed"):
        assert out[k].shape == (2, 2)


def test_eval_transform_output_size():
    x = cm.eval_transform()(Image.new("RGB", (936, 729)))
    assert tuple(x.shape) == (3, cm.IMG_H, cm.IMG_W)


# ----------------------------------------------------------------------------- crop + panel check
def test_full_screenshot_is_cropped_to_training_panel_size():
    panel, how = core.to_panel(Image.new("RGB", (2880, 1800)))
    assert how == "cropped_screenshot"
    assert panel.size == (936, 729)


def test_panel_crop_is_used_as_is():
    img = Image.new("RGB", (936, 729))
    panel, how = core.to_panel(img)
    assert how == "already_panel" and panel is img


def test_unexpected_shape_is_flagged():
    assert core.to_panel(Image.new("RGB", (500, 500)))[1] == "unknown_shape"


def _ref_gray():
    need(REFERENCE_PANEL, "reference_panel.png")
    g = core._small_gray(Image.open(REFERENCE_PANEL))
    return g - g.mean()


def test_panel_similarity_identical_image_is_high():
    assert core.panel_similarity(Image.open(REFERENCE_PANEL), _ref_gray()) > 0.99


def test_panel_similarity_blank_image_triggers_warning():
    assert core.panel_similarity(Image.new("RGB", (936, 729), "white"), _ref_gray()) < core.PANEL_SIM_WARN


# ----------------------------------------------------------------------------- answer templates
P = dict(altitude=5518.0, airspeed=110.0, heading=5.0, pitch=1.3, roll=0.4, vs=60.0, rpm=2410.0)


def ans(intent, phase="level_flight", score=1.0, **over):
    return core.make_answer(intent, dict(P, **over), phase, 0.99, score)


@pytest.mark.parametrize("intent, score", [(None, 1.0), ("get_altitude", 0.2)])
def test_out_of_scope_or_weak_routing_gets_not_applicable(intent, score):
    assert ans(intent, score=score) == core.NOT_APPLICABLE


def test_altitude_answer():
    assert "5,518 ft" in ans("get_altitude")


def test_heading_has_three_digits():
    assert "005°" in ans("get_heading")


@pytest.mark.parametrize("roll, expect", [(1.9, "level"), (-1.9, "level"), (-10, "left"), (12, "right")])
def test_bank_threshold_two_degrees(roll, expect):
    assert expect in ans("is_banking", roll=roll)


@pytest.mark.parametrize("phase, vs, expect", [
    ("taxi", 0, "on the ground"), ("engine_off", 0, "on the ground"),
    ("takeoff_roll", 0, "takeoff roll"), ("climb", 600, "Yes"),
    ("level_flight", -300, "descending"), ("level_flight", 50, "roughly level")])
def test_climbing_answer_uses_phase_and_200_fpm_threshold(phase, vs, expect):
    assert expect in ans("is_climbing", phase=phase, vs=vs)


@pytest.mark.parametrize("rpm, expect", [(150, "not running"), (2410, "2,410 RPM")])
def test_rpm_engine_off_threshold(rpm, expect):
    assert expect in ans("get_rpm", rpm=rpm)


def test_low_airspeed_gets_scale_warning():
    assert "no markings below" in ans("get_airspeed", airspeed=20)
    assert "no markings" not in ans("get_airspeed", airspeed=110)


# ----------------------------------------------------------------------------- question routing
@pytest.mark.parametrize("question, intent", [
    ("What is the vertical speed?", "get_vertical_speed"),     # must not be taken as airspeed
    ("What's the engine speed?", "get_rpm"),                   # must not be taken as airspeed
    ("How high are we?", "get_altitude"),
    ("How fast are we going?", "get_airspeed"),
    ("Which direction are we heading?", "get_heading"),
    ("Are the wings level?", "is_banking"),
    ("Are we climbing?", "is_climbing"),
    ("What phase of flight is this?", "flight_phase"),
    ("What's the weather like today?", None),
])
def test_keyword_fallback_router(question, intent):
    assert core.CopilotSystemV2.keyword_fallback(question) == intent


@pytest.fixture(scope="module")
def tfidf_router():
    pytest.importorskip("sklearn")
    r = object.__new__(core.CopilotSystemV2)    # router only -- no model loaded
    r._init_router(use_llm=False)
    assert r.have_tfidf
    return r


@pytest.mark.parametrize("question, intent", [
    ("what altitude are we flying at", "get_altitude"),
    ("how many knots", "get_airspeed"),
    ("what is our compass heading", "get_heading"),
    ("is the engine running", "get_rpm"),
])
def test_tfidf_router(tfidf_router, question, intent):
    label, score, _ = tfidf_router.route_question(question)
    assert label == intent and score > 0


def test_tfidf_router_rejects_unrelated_question(tfidf_router):
    label, score, _ = tfidf_router.route_question("tell me a joke")
    assert label is None and score == 0
