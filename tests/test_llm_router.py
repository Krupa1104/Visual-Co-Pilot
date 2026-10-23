"""INTEGRATION tests -- Qwen2.5-1.5B question router (cockpit). Marker `llm` (slowest:
loads the ~3 GB model; each question takes seconds on CPU).

Questions are worded differently from the router's examples on purpose, to check it
understands paraphrases, plus off-topic questions that must be rejected.
"""
import os
import time

import pytest

from conftest import MODEL_DIR, REFERENCE_PANEL

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def router():
    pytest.importorskip("transformers")
    from copilot_core_v2 import CopilotSystemV2
    if not os.path.exists(os.path.join(MODEL_DIR, "best_model.pt")):
        pytest.skip("cockpit model not found")
    c = CopilotSystemV2(model_dir=MODEL_DIR, reference_panel=REFERENCE_PANEL, use_llm=True)
    if not c.have_llm_gate:
        pytest.skip("Qwen could not be loaded (see terminal warning)")
    return c


@pytest.mark.parametrize("question, intent", [
    ("How high is the plane flying right now?", "get_altitude"),
    ("What's our height above sea level?", "get_altitude"),
    ("What speed is the aircraft doing?", "get_airspeed"),
    ("Which way is the nose pointing on the compass?", "get_heading"),
    ("Is the aircraft tilted to one side?", "is_banking"),
    ("Is the plane going up?", "is_climbing"),
    ("How quickly are we gaining or losing height, in feet per minute?", "get_vertical_speed"),
    ("Is the engine on?", "get_rpm"),
    ("Are we on the ground or flying?", "flight_phase"),
    ("What's the capital of France?", None),
    ("Tell me a joke", None),
])
def test_qwen_routes_paraphrases(router, question, intent):
    t0 = time.perf_counter()
    label, conf, _ = router.route_question(question)
    print(f"{question!r} -> {label} ({(time.perf_counter() - t0):.1f} s)")
    assert label == intent
