"""Shared setup for the Visual Co-Pilot test suite.

Run from the REPO ROOT (the folder with integrated.py):
    python -m pytest -m "not slow and not llm" -v      # unit tests, no model files, seconds
    python -m pytest -m "slow" -v                       # integration + performance, needs models
    python -m pytest -v --html=tests/results/report.html --self-contained-html   # everything + report

Paths can be overridden with environment variables:
    VCP_RADAR_BASE   radar_base folder   (default: radar/radar_base, then radar_base)
    VCP_YOLO_ROOT    YOLOv5 code folder  (default: radar/yolov5)
    VCP_HOLDOUT      cockpit hold-out set (default: cockpit2/dataset_holdout)
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCKPIT2_GUI = os.path.join(REPO, "cockpit2", "gui")
RADAR_DIR = os.path.join(REPO, "radar")
for p in (COCKPIT2_GUI, RADAR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS_DIR = os.path.join(REPO, "tests", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_DIR = os.path.join(REPO, "cockpit2", "dataset_v2", "model_v2")
REFERENCE_PANEL = os.path.join(REPO, "cockpit2", "dataset_v2", "reference_panel.png")
HOLDOUT_DIR = os.environ.get("VCP_HOLDOUT", os.path.join(REPO, "cockpit2", "dataset_holdout"))


def _first_existing(*paths):
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return paths[0]


RADAR_BASE = _first_existing(os.environ.get("VCP_RADAR_BASE"),
                             os.path.join(REPO, "radar", "radar_base"),
                             os.path.join(REPO, "radar_base"))
YOLO_ROOT = os.environ.get("VCP_YOLO_ROOT", os.path.join(REPO, "radar", "yolov5"))

# Aviation tolerances used for every cockpit accuracy number in the project (train_v2 / eval_v2).
TOL = {"pitch": 1.0, "roll": 2.0, "airspeed": 5.0, "altitude": 100.0, "heading": 5.0,
       "vs": 100.0, "turn": 0.2, "rpm": 100.0}


def need(path, what):
    if not os.path.exists(path):
        pytest.skip(f"{what} not found at {path}")


# ----------------------------------------------------------------------------- shared samples / models
def holdout_sample(per_phase=5, seed=0):
    """5 hold-out images per phase (fixed seed -> the same 25 images every run)."""
    import pandas as pd
    path = os.path.join(HOLDOUT_DIR, "labels.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path).drop_duplicates(["phase", "image"])
    rows = []
    for _, g in df.groupby("phase"):
        rows += g.sample(min(per_phase, len(g)), random_state=seed).to_dict("records")
    return rows


HOLDOUT = holdout_sample()


@pytest.fixture(scope="session")
def cockpit():
    """The real cockpit system, TF-IDF router (no Qwen download needed)."""
    if not os.path.exists(os.path.join(MODEL_DIR, "best_model.pt")):
        pytest.skip(f"cockpit model not found in {MODEL_DIR}")
    from copilot_core_v2 import CopilotSystemV2
    return CopilotSystemV2(model_dir=MODEL_DIR, reference_panel=REFERENCE_PANEL, use_llm=False)
