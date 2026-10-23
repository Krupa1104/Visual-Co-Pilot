"""PERFORMANCE tests (non-functional) -- latency of each cockpit stage. Marked `slow`.

Measures, on this machine, over the hold-out sample (25 images):
  preprocess (crop + panel check + resize), model inference, TF-IDF routing.
Results are written to tests/results/performance.json with the device used, so the
report can state exactly what was measured (CPU vs GPU). Limits are loose sanity
bounds, not targets.
"""
import json
import os
import platform
import statistics
import time

import pytest
import torch
from PIL import Image

from conftest import HOLDOUT, HOLDOUT_DIR, RESULTS_DIR

pytestmark = pytest.mark.slow


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def test_cockpit_latency(cockpit):
    if not HOLDOUT:
        pytest.skip("no hold-out images")
    imgs = [Image.open(os.path.join(HOLDOUT_DIR, r["phase"], r["image"])).convert("RGB") for r in HOLDOUT]
    cockpit.run_perception(imgs[0])                       # warm-up (first call is slower)
    pre, inf, route = [], [], []
    for img in imgs:
        _, info = cockpit.run_perception(img)
        pre.append(info["preprocess_ms"])
        inf.append(info["inference_ms"])
        t0 = time.perf_counter()
        cockpit.route_question("What is the current altitude?")
        route.append((time.perf_counter() - t0) * 1000)

    def summary(v):
        return {"median_ms": round(statistics.median(v), 1), "p95_ms": round(pct(v, 95), 1),
                "max_ms": round(max(v), 1)}

    report = {"device": str(cockpit.device), "cuda_available": torch.cuda.is_available(),
              "torch": torch.__version__, "cpu": platform.processor(), "n_images": len(imgs),
              "router": "TF-IDF (use_llm=False)",
              "preprocess": summary(pre), "inference": summary(inf), "routing": summary(route)}
    with open(os.path.join(RESULTS_DIR, "performance.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    assert report["inference"]["p95_ms"] < 5000          # sanity bound for CPU
    assert report["preprocess"]["p95_ms"] < 2000
