# Visual Co-Pilot

Visual question answering for aircraft instruments and ATC radar. Upload a cockpit instrument
panel or a radar display, ask a question in plain English (typed or spoken), and get an answer
read from the image. One Gradio app (`integrated.py`) hosts both modules.

PES University capstone project. Team: Krupa R, Madhumitha N V, Krithika A, M Asritha.
Guide: Dr Vandana M L.

| Module | What it reads | Models | Details |
|---|---|---|---|
| Cockpit (`cockpit2/`) | Cessna 172 panel in FlightGear: altitude, airspeed, heading, pitch, roll, vertical speed, turn rate, engine RPM, flight phase | CockpitNetV2 (ResNet18 with regression and needle-angle heads) | [`cockpit2/README.md`](cockpit2/README.md) |
| Radar (`radar2/`) | Synthetic ATC radar images: aircraft count, quadrant/position, altitude, heading, callsign | YOLOv5 aircraft detector, PaddleOCR for the aircraft labels (PP-OCRv5 mobile by default, PP-OCRv6 medium with `--radar_ocr_models medium`), radar VQA classifier for the remaining questions | `radar2/radar_system.py`, `radar2/scripts/radar_eval.py` |

How questions are answered:
- Cockpit: rules first (a question that matches exactly one instrument by keywords), Qwen2.5-1.5B-Instruct
  only for the rest, keyword fallback if Qwen can't be used. Questions that name only things the model
  does not read (fuel, oil, gear, radio...) are answered "not in scope".
- Radar: keyword relevance check, then rules: count and quadrant from YOLO boxes; callsign, altitude and
  heading from OCR of the label next to each detected aircraft; the VQA classifier for the rest.

## Repository layout (branch `integrate`)
```
integrated.py      # the app: landing page + Cockpit and Radar panels (run this)
cockpit2/          # cockpit module (current): model, training, evaluation, results, question router
radar2/            # radar module (current): radar_system.py, src/, configs/, scripts/, test split, eval results
cockpit/           # cockpit module v1 (kept for comparison only)
radar/             # previous radar module (kept in the repo, no longer used by the app)
tests/, pytest.ini # test suite (unit, integration, performance, LLM router)
```
Other branches: `vqa-cockpit` (cockpit only), `vqa-radar` (previous radar module), `Radar_Phase1`
(early radar work), `main` (initial commit).

## 1. What is NOT in git (ask the team for these)
Model weights, datasets and images are not committed. Put them exactly where shown and the app finds
them without extra arguments.

| What | Needed for | Put it at | Size | Ask |
|---|---|---|---|---|
| Cockpit model `best_model.pt` (run 6) | Cockpit panel | `cockpit2\dataset_v2\model_v2\best_model.pt` | ~45 MB | Krupa |
| Radar `models` folder: `vqa\best.pt` (VQA classifier) and `yolov5\` (YOLOv5 code with the detector weights in `yolov5\weights\best.pt`) | Radar panel | `radar2\models\` | ~560 MB | Krupa / Madhumitha |
| Radar images (`radar2\data\synthetic_complete\images\`) | trying the radar panel, radar integration tests, `radar_eval.py` | `radar2\data\synthetic_complete\` | – | Krupa / Madhumitha |
| Cockpit demo images (`dataset_v2_test.zip`, 375 panel crops) | trying the cockpit panel | anywhere | small | Krupa |
| Cockpit hold-out set `dataset_holdout\` (500 images + labels.csv) | slow cockpit integration tests | `cockpit2\dataset_holdout\` | – | Krupa |

The radar test split (which images and questions are test data) is in git: `radar2\data\splits\test.jsonl`
and `test_frames.json`. The YOLOv5 copy used by the radar team is `v7.0-561-gaf08ba15` (upstream commit
`af08ba1` of https://github.com/ultralytics/yolov5). PaddleOCR downloads its OCR models the first time the
radar module loads (internet needed once).

## 2. Setup (Windows, once)
Python 3.12 (tested with 3.12.4). From the folder where you want the project:
```
git clone -b integrate https://github.com/Krupa1104/Visual-Co-Pilot.git
cd Visual-Co-Pilot
python -m venv venv_final
venv_final\Scripts\activate
python -m pip install --upgrade pip
pip install -r radar2\requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers scikit-learn pytest pytest-html
```
- `radar2\requirements.txt` is the radar team's pinned environment (CPU build of torch 2.15.0, Gradio 6,
  PaddleOCR 3.7, PaddlePaddle 3.3, ultralytics). The extra index is needed for the `+cpu` torch wheels.
- `transformers` (Qwen router) and `scikit-learn` (TF-IDF fallback) are needed by the cockpit module.
- Everything runs on CPU; no GPU is needed.
- Then copy in the files from section 1.

## 3. Run
From the repository root:
```
python integrated.py
```
Open the local URL it prints (usually http://127.0.0.1:7860) and choose a module. A model loads the
first time its card is opened.

- The first cockpit question that the rules can't route downloads Qwen2.5-1.5B-Instruct from Hugging Face
  (about 3 GB, cached afterwards). `--cockpit_no_llm` skips Qwen (rules + TF-IDF only).
- `--radar_ocr_models medium`: larger OCR models (slightly more accurate on altitude/heading, about 3x
  slower; see section 5). `--radar_ocr_mode full`: OCR on the whole image instead of crops around aircraft.
- `--cockpit_no_rules`: send every cockpit question to Qwen (for comparison only).
- Paths can be changed with `--cockpit2_model_dir`, `--radar_base_dir`, `--radar_yolo_root`; see
  `python integrated.py --help`.
- Voice input (🎤 Speak) uses the browser's speech recognition: Chrome or Edge, and Chrome needs an internet
  connection for it. Read-aloud works offline.

**Which images to use**
- Cockpit: a full FlightGear screenshot taken with the capture setup (cropped automatically) or an
  instrument-panel crop, e.g. from `dataset_v2_test.zip`. Other aircraft, views or resolutions are out of
  scope; the app shows a panel-match warning for them.
- Radar: images from `radar2\data\synthetic_complete\images\`; for a fair demo use test images (listed in
  `radar2\data\splits\test_frames.json`).

## 4. Tests
```
python -m pytest -m "not slow and not llm" -v
python -m pytest -v -rA
```
The first command runs the unit tests (no model files needed). The second runs everything; the slow tests
need the files from section 1 and the `llm` tests need Qwen. Tests whose files are missing are skipped.

Last full run (5 Nov 2026, CPU): 198 passed, 4 failed, 1 skipped (141 s). The 4 failures are measured
limitations, kept as they are:
- 2 in `test_llm_router.py` (Qwen alone, without rules): "Which way is the nose pointing on the compass?"
  and "Is the engine on?" are routed to the wrong intent. In the app the rules answer both correctly.
- 2 in `test_radar2_integration.py`: quadrant questions where OCR "could not find a readable label" next to
  the aircraft.
The old `radar/` module's tests skip without its data.

## 5. Results (short)
**Cockpit gauge reading**, final model (run 6), fresh hold-out set of 500 frames evaluated once: 99.8% of
answers within aviation tolerance (4492/4500), 98.6% of images with all 9 answers correct, flight phase
100%. Details and limitations: [`cockpit2/README.md`](cockpit2/README.md).

**Cockpit question routing**, blind set 2 (60 questions written by Gemini without seeing the rules;
`cockpit2/results/router_eval/router_questions_blind2/`): rules then Qwen 52/60 (86.7%) vs Qwen alone
49/60 (81.7%); 38/60 answered by rules, all correct; median routing time 0.10 ms vs 4.7 s on CPU.

**Radar**, `radar2/scripts/radar_eval.py`, 120 questions from test images (20 per type, seed 2),
default settings (mobile OCR models on crops), CPU (`radar2/outputs/radar_eval/mobile_crops_n20_seed2/`):

| Count | Altitude | Heading | Quadrant | Yes/no | Callsign | All | Median time |
|---|---|---|---|---|---|---|---|
| 19/20 | 12/20 | 13/20 | 18/20 | 20/20 | 18/20 (lenient) | 83/120 (69.2%) | 3.5 s/question |

- Medium OCR models on crops: 86/120 (71.7%) at 10.8 s/question. The rule set before the run
  (better on altitude+heading AND under 10 s) was not met, so mobile stays the default.
- The radar module as received before the speed work (medium models, full image): 31/60 at 28.9 s/question
  on an earlier 60-question sample (seed 0).
- Yes/no: 19 of the 20 sampled questions had the answer "no" (about 75% of the dataset's yes/no answers
  are "no"), so 20/20 is weaker evidence than it looks.
- Callsign is scored leniently (correct callsign among those listed): the "selected aircraft" in these
  questions is not marked in the image.

## 6. Known limitations
- Cockpit: in-distribution only (FlightGear Cessna 172, one camera view and screen resolution).
- Cockpit routing: some heading and engine paraphrases are misrouted; off-topic questions that use a
  cockpit word are accepted by the rules.
- Radar: the main error is OCR not finding a readable label next to a detected aircraft (altitude,
  heading, quadrant-by-callsign questions); YOLO occasionally miscounts by one; the YOLO size and
  confidence settings were configured, not tuned on validation data.
- Model weights and datasets have to be shared outside git (section 1).