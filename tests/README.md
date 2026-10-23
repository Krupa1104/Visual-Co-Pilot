# Visual Co-Pilot test suite

Copy `tests\` and `pytest.ini` into the repo root (the folder with `integrated.py`), then in the app's venv:

    pip install pytest pytest-html

| Command | What runs | Needs |
|---|---|---|
| `python -m pytest -m "not slow and not llm" -v` | Unit tests: cockpit decoding/crop/answers/routing, radar routing (fake YOLO), app callbacks | nothing extra, seconds |
| `python -m pytest -m slow -v` | Integration (real models: 25 hold-out images, question->answer, screenshot crop, 20 radar frames) + performance | model files, minutes |
| `python -m pytest -m llm -v` | Qwen router on paraphrased / off-topic questions | Qwen (downloaded on first run) |
| `python -m pytest -v --html=tests/results/report.html --self-contained-html` | Everything + an HTML report for the PPT | all of the above |

Outputs: `tests/results/report.html`, `tests/results/performance.json` (records CPU vs GPU).
Paths can be changed with `VCP_RADAR_BASE`, `VCP_YOLO_ROOT`, `VCP_HOLDOUT`.
