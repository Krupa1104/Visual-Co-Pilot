# Visual Co-Pilot

Visual question answering for aircraft instruments and ATC radar. Upload a cockpit instrument
panel or a radar display, ask a question in plain English (typed or spoken), and get an answer
read from the image. One Gradio app (`integrated.py`) hosts both modules.

PES University capstone project. Team: Krupa R, Madhumitha N V, Krithika A, M Asritha.
Guide: Dr Vandana M L.

| Module | What it reads | Main model | Details |
|---|---|---|---|
| Cockpit (v2) | Cessna 172 panel in FlightGear: altitude, airspeed, heading, pitch, roll, vertical speed, turn rate, engine RPM, flight phase | CockpitNetV2 (ResNet18 with regression and needle-angle heads) | [`cockpit2/README.md`](cockpit2/README.md) |
| Radar | Synthetic ATC radar images: aircraft count, position/quadrant, altitude, heading, callsign | ResNet50 CNN VQA model + YOLOv5 aircraft detector | [`radar/README.md`](radar/README.md) |

Questions are routed to an answer type by Qwen2.5-1.5B-Instruct (cockpit intent router and radar
relevance gate), with TF-IDF / keyword fallbacks.

## Repository layout (branch `integrate`)
```
integrated.py      # the app: landing page + Cockpit and Radar panels (run this)
cockpit2/          # cockpit module v2 (current): model, training, evaluation, results
cockpit/           # cockpit module v1 (kept for comparison only)
radar/             # radar module: radar_system.py (used by the app), training code, notebooks
tests/, pytest.ini # test suite (unit, integration, performance, LLM router)
```
Other branches: `vqa-cockpit` (cockpit only, same `cockpit2/` as here), `vqa-radar` (radar only,
same `radar/` as here), `Radar_Phase1` (early radar work), `main` (initial commit).

## 1. What is NOT in git (ask the team for these)
Model weights and datasets are too large for git. Put them exactly where shown, and the app finds
them without extra arguments.

| What | Needed for | Put it at | Size | Ask |
|---|---|---|---|---|
| Cockpit model `best_model.pt` (run 6) | Cockpit panel | `cockpit2\dataset_v2\model_v2\best_model.pt` | ~45 MB | Krupa |
| `radar_base` folder | Radar panel | `radar\radar_base\` | ~250 MB | Madhumitha |
| Cockpit demo images (`dataset_v2_test.zip`, 375 panel crops) | trying the cockpit panel | anywhere | small | Krupa |
| Cockpit hold-out set `dataset_holdout\` (500 images + labels.csv) | slow integration tests only | `cockpit2\dataset_holdout\` | – | Krupa |

`radar\radar_base\` must contain at least:
```
models\cnn_best_model.pth                     # radar CNN
models\yolov5_aircraft\weights\best.pt        # YOLOv5 aircraft detector
data\processed\answer_index.json              # CNN answer classes
data\raw\rendered_frames.json                 # per-frame metadata (callsigns, altitudes, positions)
data\raw\images\                              # the 2,000 radar images (also the demo images)
```
`logs\` is created by the app. The rest of the folder (training json files, `data\processed\images`,
training curves) is only needed to retrain.

Not needed to run the app: the full cockpit dataset `dataset_v2` (2,500 images, retraining only)
and the v1 `consolidated\` folder (only for `--cockpit_version v1`).

## 2. Setup (Windows, once)
Python 3.10 or newer (Gradio 6 needs it). From the folder where you want the project:
```
git clone -b integrate https://github.com/Krupa1104/Visual-Co-Pilot.git
cd Visual-Co-Pilot
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```
1. Install PyTorch first, with the command for your GPU/CUDA version from
   https://pytorch.org/get-started/locally/ (the CPU-only build also works, just slower).
2. Install the rest:
   ```
   pip install -r cockpit2\requirements.txt -r radar\requirements-radar.txt gradio
   ```
3. YOLOv5 code (not part of this repo; it is its own repository). Clone it into `radar\yolov5` and
   use the same commit the radar detector was run with:
   ```
   git clone https://github.com/ultralytics/yolov5 radar\yolov5
   git -C radar\yolov5 checkout 402e17ddf820996f51a191cbb798376e1144f069
   pip install -r radar\yolov5\requirements.txt
   ```
4. Copy in the files from section 1.

## 3. Run
From the repository root:
```
python integrated.py
```
Open the local URL it prints (usually http://127.0.0.1:7860) and choose a module. A model loads
the first time its card is opened.

- **First run downloads Qwen2.5-1.5B-Instruct** from Hugging Face (about 3 GB, cached afterwards).
  The radar module reuses the cockpit's copy if the cockpit was opened first.
- `--cockpit_no_llm`: cockpit uses the TF-IDF router instead of Qwen (fast start, no download).
- `--cockpit_llm_on_cpu`: run the Qwen router on CPU if the GPU runs out of memory.
- `--radar_base_dir`, `--radar_yolo_root`, `--cockpit2_model_dir`: only if your files are
  somewhere else. See `python integrated.py --help`.
- Voice input (🎤 Speak) uses the browser's speech recognition: Chrome or Edge, and Chrome needs
  an internet connection for it. Read-aloud works offline.

**Which images to use**
- Cockpit: a full FlightGear screenshot taken with the capture setup (cropped automatically) or an
  instrument-panel crop, e.g. from `dataset_v2_test.zip`. Other aircraft, views or resolutions are
  out of scope; the app shows a panel-match warning for them.
- Radar: images from `radar_base\data\raw\images\`. Callsign and altitude answers come from
  `rendered_frames.json` and only work for these frames (they are matched by file name).

## 4. Tests
```
pip install pytest pytest-html
python -m pytest -m "not slow and not llm" -v
python -m pytest -v --html=tests/results/report.html --self-contained-html
```
The first command runs the unit tests (no model files needed, seconds). The second runs
everything and writes an HTML report: the slow tests need the files from section 1, the `llm`
tests need Qwen. Tests whose files are missing are skipped, not failed. See `tests/README.md`.

Last full run (23 Oct 2026, CPU): 132 tests, 124 passed, 8 failed. The 8 failures are measured
model limitations, kept as they are: 6 radar frames where YOLOv5's aircraft count is off by one
(14/20 exact), and 2 paraphrased cockpit questions that Qwen routes to the wrong intent (9/11).

## 5. Results (short)
Cockpit v2, final model (run 6), on a fresh hold-out set of 500 frames evaluated once:
99.8% of answers within aviation tolerance (4492/4500), 98.6% of images with all 9 answers
correct, flight phase 100%. Per-gauge numbers, baselines, training history and limitations are in
[`cockpit2/README.md`](cockpit2/README.md). These results are in-distribution (same simulator,
aircraft, camera view and resolution as training).

## 6. Known limitations
- Cockpit: in-distribution only (FlightGear Cessna 172, one camera view and screen resolution).
- Radar: YOLOv5 counts are off by one on some frames; metadata-based answers need frames from
  `rendered_frames.json`.
- Qwen routing can misroute unusual phrasings; the standard sample questions route correctly.
- `best_model.pt`, `radar_base` and the datasets have to be shared outside git (section 1).
