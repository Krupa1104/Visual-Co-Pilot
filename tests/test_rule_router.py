"""UNIT tests -- cockpit rule router (copilot_core_v2.rule_route), the fast path in front of Qwen.
No model files needed. These are behaviour checks of the rules (what they answer, what they leave to
Qwen); the accuracy comparison against Qwen is cockpit2/process/eval_router.py, not this file.
"""
import pytest

import copilot_core_v2 as core


@pytest.mark.parametrize("question, intent", [
    # the app's sample questions must all take the fast path
    ("What is the current altitude?", "get_altitude"),
    ("How fast are we going?", "get_airspeed"),
    ("What heading are we on?", "get_heading"),
    ("Are we banking left or right?", "is_banking"),
    ("Are we climbing?", "is_climbing"),
    ("What phase of flight are we in?", "flight_phase"),
    ("What is the vertical speed?", "get_vertical_speed"),
    ("What is the engine RPM?", "get_rpm"),
    # a more specific phrase wins over the general word inside it
    ("What is the rate of climb?", "get_vertical_speed"),
    ("Are we gaining altitude?", "is_climbing"),
    ("Is the nose pitched up?", "is_climbing"),
    ("Is the propeller turning?", "get_rpm"),
])
def test_rules_answer_clear_questions(question, intent):
    assert core.rule_route(question) == intent


@pytest.mark.parametrize("question", [
    "Is the plane going up?",                 # no rule word -> Qwen
    "What is the speed of sound?",            # bare "speed" is not a rule
    "What did I have for breakfast?",         # whole words only: "fast" inside "breakfast"
    "When do we return home?",                # whole words only: "turn" inside "return"
    "Are we climbing and what is our heading?",  # two intents -> Qwen
    "Tell me a joke",
    "",
])
def test_rules_defer_to_qwen(question):
    assert core.rule_route(question) is None


@pytest.mark.parametrize("question", [
    "How much fuel do we have?",
    "What is the oil temperature?",
    "Are the flaps down?",
    "Which radio frequency are we on?",
    "Who is flying, the pilot or the co-pilot?",
])
def test_rules_reject_unsupported_cockpit_items(question):
    assert core.rule_route(question) == core.OUT_OF_SCOPE


def test_supported_reading_wins_over_unsupported_word():
    # names a supported reading and an unsupported item -> not decided by rules, left to Qwen
    assert core.rule_route("What is our altitude and how much fuel is left?") is None
