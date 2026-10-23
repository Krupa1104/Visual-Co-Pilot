r"""
integrated.py
Single Gradio app for the Visual Co-Pilot project. Replaces the earlier
Tkinter-launcher-that-spawns-a-subprocess design (fragile: guessed a fixed
wait time before opening the browser, and could hit a connection-refused
page if the model hadn't finished loading yet).

One process, one URL, one thing to wait on (Gradio's own "Running on
local URL" message) -- same as running cockpit/gui/app.py or radar/app.py
directly today.

FLOW:
  1. You run this once, with startup args for whichever module(s) you plan
     to use (see --help). Nothing is loaded yet at this point.
  2. One browser tab opens to a landing view: two module cards, Cockpit and Radar.
  3. Clicking a card switches to that module's panel and LOADS its system
     (CopilotSystemV2 / CopilotSystem or RadarSystem) right away, the first
     time only, with a status pill showing "Loading model..." -> "Model ready".
     (Before the GUI redesign the model only loaded on the first Run Analysis,
     so the first question looked frozen.)
  4. Both panels share one layout: input (image, question, voice) on the left;
     answer, readings and a collapsible "Details" section on the right.
  5. "<- Modules" returns to the landing view. Systems stay loaded in memory.

GUI redesign merged onto the team's radar-voice version of this file (radar Speak button, read-aloud, radar_base / radar/yolov5 defaults). Every GUI change is in this file;
the cockpit model code (cockpit2/gui/copilot_core_v2.py, cockpit/gui/copilot_core.py) is called exactly as before. radar/radar_system.py is maintained by the radar team;
its final version (routing fixes, YOLO average detection confidence) is committed alongside this file.

Requires Gradio 6 (theme/css/js are passed to launch(), not to gr.Blocks()).

Setup:
  pip install gradio torch torchvision pandas pillow sentence-transformers --break-system-packages
  (plus radar/requirements-radar.txt's deps, e.g. YOLOv5's requirements --
   both modules' dependencies now need to coexist in ONE Python environment,
   since this is one process, not two separate subprocesses like before)

Run FROM THE REPO ROOT (needs cockpit2/ (and/or cockpit/) and radar/ next to it):
  python integrated.py        (expects radar/radar_base and radar/yolov5)
      -> Cockpit uses model v2 by default (cockpit2/dataset_v2/model_v2/best_model.pt)
  python integrated.py --cockpit_version v1 --cockpit_data_dir "C:\...\consolidated" ...
      -> old v1 model (kept for comparison only)

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

If you only plan to use one module this run, you can omit the other
module's required arg -- that card will just show a clear error (not a
crash) if you click it without its data configured.
"""

import os
import sys
import time
import threading
import html
import argparse

import gradio as gr
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
COCKPIT_DIR = os.path.join(REPO_ROOT, "cockpit", "gui")
COCKPIT2_DIR = os.path.join(REPO_ROOT, "cockpit2", "gui")
RADAR_DIR = os.path.join(REPO_ROOT, "radar")

# Both modules do local, same-folder imports (`from copilot_core import
# CopilotSystem`, `from radar_system import RadarSystem`). Since everything
# now runs in ONE process (unlike the old subprocess-per-module launcher),
# both folders need to be on sys.path so those local imports resolve no
# matter which one loads first.
for _p in (COCKPIT_DIR, COCKPIT2_DIR, RADAR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

parser = argparse.ArgumentParser()
# Cockpit model choice. v2 = CockpitNetV2 on dataset_v2 (default); v1 = old CockpitNet.
parser.add_argument("--cockpit_version", choices=["v1", "v2"], default="v2")
parser.add_argument("--cockpit2_model_dir",
                    default=os.path.join(REPO_ROOT, "cockpit2", "dataset_v2", "model_v2"),
                    help="v2: folder containing best_model.pt")
parser.add_argument("--cockpit2_reference",
                    default=os.path.join(REPO_ROOT, "cockpit2", "dataset_v2", "reference_panel.png"),
                    help="v2: reference panel image for the 'is this the training view?' check")
parser.add_argument("--cockpit_llm_on_cpu", action="store_true",
                    help="v2: run the Qwen intent router on CPU (use if the GPU runs out of memory)")
parser.add_argument("--cockpit_no_llm", action="store_true",
                    help="v2: skip the Qwen router, use the TF-IDF fallback (fast start, no download)")
# v1 cockpit args -- same names/defaults as cockpit/gui/app.py's own parser.
parser.add_argument("--cockpit_data_dir", default=None,
                     help="v1 only: your local consolidated/ folder.")
parser.add_argument("--cockpit_img_width", type=int, default=224)
parser.add_argument("--cockpit_img_height", type=int, default=224)
parser.add_argument("--cockpit_use_letterbox", action="store_true")
parser.add_argument("--cockpit_confidence_threshold", type=float, default=0.35)
# Radar args -- same names/defaults as radar/app.py's own parser.
parser.add_argument("--radar_base_dir", default=os.path.join(REPO_ROOT, "radar", "radar_base"),
                     help="Your local radar_base/ folder (models/, data/, logs/). "
                          "Default: radar/radar_base.")
parser.add_argument("--radar_repo_root", default=None)
parser.add_argument("--radar_yolo_root", default=os.path.join(REPO_ROOT, "radar", "yolov5"),
                     help="Local YOLOv5 code. Default: radar/yolov5.")
parser.add_argument("--radar_yolo_conf", type=float, default=0.25)
parser.add_argument("--radar_cnn_checkpoint", default="cnn_best_model.pth")
parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
args = parser.parse_args()

# =========================================================================
# LAZY LOADING -- a module's system is loaded the first time its card is
# opened (or, as a fallback, on its first Run Analysis).
# =========================================================================
_cockpit_system = None
_radar_system = None
# One lock per module: if Run Analysis is clicked while the card's load is still running, the second
# call waits for the first load instead of starting a second copy of the model.
_cockpit_lock = threading.Lock()
_radar_lock = threading.Lock()


def get_cockpit_system():
    with _cockpit_lock:
        return _get_cockpit_system()


def _get_cockpit_system():
    global _cockpit_system
    if _cockpit_system is not None:
        return _cockpit_system, None
    if args.cockpit_version == "v2":
        ckpt = os.path.join(args.cockpit2_model_dir, "best_model.pt")
        if not os.path.exists(ckpt):
            return None, (f"Cockpit v2 model not found at {ckpt} -- train it (cockpit2/process/"
                           f"train_v2.py) or pass --cockpit2_model_dir.")
        from copilot_core_v2 import CopilotSystemV2
        print(f"Loading Cockpit v2 from {ckpt} ...")
        _cockpit_system = CopilotSystemV2(
            model_dir=args.cockpit2_model_dir,
            reference_panel=args.cockpit2_reference,
            confidence_threshold=args.cockpit_confidence_threshold,
            llm_on_cpu=args.cockpit_llm_on_cpu,
            use_llm=not args.cockpit_no_llm,
        )
        print(f"Cockpit v2 loaded: {_cockpit_system.model_info}")
        return _cockpit_system, None
    if not args.cockpit_data_dir:
        return None, ("Cockpit isn't configured for this run -- restart with "
                       "--cockpit_data_dir pointing at your consolidated/ folder.")
    from copilot_core import CopilotSystem
    print("Loading Cockpit (CopilotSystem)...")
    _cockpit_system = CopilotSystem(
        data_dir=args.cockpit_data_dir,
        img_width=args.cockpit_img_width,
        img_height=args.cockpit_img_height,
        use_letterbox=args.cockpit_use_letterbox,
        confidence_threshold=args.cockpit_confidence_threshold,
    )
    print(f"Cockpit loaded (checkpoint epoch {_cockpit_system.checkpoint_epoch}).")
    return _cockpit_system, None


def get_radar_system():
    with _radar_lock:
        return _get_radar_system()


def _get_radar_system():
    global _radar_system
    if _radar_system is not None:
        return _radar_system, None
    if not args.radar_base_dir:
        return None, ("Radar isn't configured for this run -- restart with "
                       "--radar_base_dir pointing at your radar_base/ folder.")
    from radar_system import RadarSystem
    print(f"Loading Radar (checkpoint: {args.radar_cnn_checkpoint})...")
    radar_kwargs = dict(
        base_dir=args.radar_base_dir,
        repo_root=args.radar_repo_root,
        yolo_root=args.radar_yolo_root,
        yolo_conf=args.radar_yolo_conf,
        cnn_checkpoint=args.radar_cnn_checkpoint,
    )
    shared = _shared_llm_for_radar()
    try:
        try:
            _radar_system = RadarSystem(**radar_kwargs, **shared)
            if shared:
                print("Radar is sharing the cockpit's Qwen2.5-1.5B instance (no second copy loaded).")
        except TypeError as exc:
            # Older radar_system.py versions don't accept llm_tokenizer/llm_model: load as before.
            if not shared or "llm_" not in str(exc):
                raise
            print("This radar_system.py can't share the cockpit's LLM -- loading its own copy.")
            _radar_system = RadarSystem(**radar_kwargs)
    except FileNotFoundError as exc:
        return None, f"Radar model files missing: {exc}"
    print("Radar loaded.")
    return _radar_system, None


def _shared_llm_for_radar():
    """If the cockpit is already loaded with its Qwen2.5-1.5B router, hand the same instance to the
    radar module (radar_system.RadarSystem accepts llm_tokenizer/llm_model for exactly this), so the
    process holds ONE copy of the LLM instead of two. Both modules only call generate() with their own
    prompts, so sharing doesn't change what either module answers. Only shared when the LLM sits on the
    same device the radar module sends its inputs to (radar uses cuda if available, else cpu)."""
    c = _cockpit_system
    if c is None or not getattr(c, "have_llm_gate", False) or not hasattr(c, "_llm"):
        return {}
    import torch
    radar_device = "cuda" if torch.cuda.is_available() else "cpu"
    if next(c._llm.parameters()).device.type != radar_device:
        return {}
    return {"llm_tokenizer": c._tok, "llm_model": c._llm}


# =========================================================================
# SMALL HTML HELPERS (display only)
# =========================================================================
def esc(x):
    return html.escape(str(x))


def status_pill(state, text):
    """state: idle | loading | ready | error"""
    return f'<div class="vc-status vc-status-{state}"><span class="vc-dot"></span>{esc(text)}</div>'


def chips(items):
    """items: list of (label, value, tone) with tone in {'', 'good', 'warn', 'bad'}."""
    out = []
    for label, value, tone in items:
        out.append(f'<span class="vc-chip {tone}"><span class="vc-chip-label">{esc(label)}</span>'
                   f'{esc(value)}</span>')
    return '<div class="vc-chips">' + "".join(out) + "</div>"


def tiles(items, active=()):
    """items: list of (key, label, value, sub). Tiles whose key is in `active` are highlighted
    (= the reading the question was about)."""
    out = []
    for key, label, value, sub in items:
        cls = "vc-tile active" if key in active else "vc-tile"
        sub_html = f'<div class="vc-tile-sub">{esc(sub)}</div>' if sub else ""
        out.append(f'<div class="{cls}"><div class="vc-tile-label">{esc(label)}</div>'
                   f'<div class="vc-tile-value">{esc(value)}</div>{sub_html}</div>')
    return '<div class="vc-tiles">' + "".join(out) + "</div>"


def kv_table(rows):
    # Plain divs, not <table>: Gradio's own table styling adds borders that the CSS can't fully override.
    body = "".join(f'<div class="vc-kv-k">{esc(k)}</div><div class="vc-kv-v">{esc(v)}</div>'
                   for k, v in rows)
    return f'<div class="vc-kv">{body}</div>'


def fmt_ms(ms):
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def tone_for(conf):
    return "good" if conf >= 0.75 else ("warn" if conf >= 0.5 else "bad")


EMPTY_READINGS = '<div class="vc-empty">Readings appear here after you run an analysis.</div>'


# =========================================================================
# COCKPIT PANEL CALLBACKS
# =========================================================================
COCKPIT_PRESET_QUESTIONS = [
    "What is the current altitude?",
    "How fast are we going?",
    "What heading are we on?",
    "Are we banking left or right?",
    "Are we climbing?",
    "What phase of flight are we in?",
]
if args.cockpit_version == "v2":
    COCKPIT_PRESET_QUESTIONS += ["What is the vertical speed?", "What is the engine RPM?"]

# Which reading tile(s) to highlight for each matched intent.
INTENT_TILES = {
    "get_altitude": ("altitude",), "get_airspeed": ("airspeed",), "get_heading": ("heading",),
    "is_banking": ("roll",), "is_climbing": ("vs", "pitch"), "get_vertical_speed": ("vs",),
    "get_rpm": ("rpm",), "flight_phase": ("phase",),
}
CROP_MODE_TEXT = {"already_panel": "instrument panel (no crop needed)"}


def cockpit_redo_voice():
    """Clear the question box so a new question can be spoken."""
    return ""


COCKPIT_OUT_OF_SCOPE = (
    "**Not in scope.** I can only answer questions about this cockpit's instruments: altitude, "
    "airspeed, heading, bank/roll, pitch/climbing, vertical speed, engine RPM and flight phase.")


def cockpit_load():
    system, err = get_cockpit_system()
    if err:
        return status_pill("error", err)
    return status_pill("ready", "Model ready")


def _cockpit_fail(msg):
    # answer, meta, readings, details, panel image (hidden), status untouched
    return msg, "", EMPTY_READINGS, "", gr.update(value=None, visible=False)


def cockpit_run_analysis(image, question):
    system, err = get_cockpit_system()
    if err:
        return _cockpit_fail(err)
    if image is None:
        return _cockpit_fail("Please upload a cockpit panel image first.")
    if not question or not question.strip():
        return _cockpit_fail("Please enter or select a question.")

    t0 = time.perf_counter()
    result = system.answer(image, question)
    wall_ms = (time.perf_counter() - t0) * 1000
    t = result["timings"]
    timing_rows = [(k.replace("_ms", "").replace("_", " ").capitalize(), fmt_ms(v))
                   for k, v in t.items() if k != "total_ms"] + [("Total", fmt_ms(wall_ms))]

    # Out-of-scope guard: the router found no cockpit intent (or too weak a match). Don't show
    # instrument readings that nobody asked for -- say plainly that the question is out of scope.
    if result["intent"] is None or result["confidence"] < args.cockpit_confidence_threshold:
        meta = chips([("Question", "out of scope", "bad"),
                      ("Routing", f"{result['confidence'] * 100:.0f}%", "bad"),
                      ("Time", fmt_ms(wall_ms), "")])
        return (COCKPIT_OUT_OF_SCOPE, meta, EMPTY_READINGS, kv_table(timing_rows),
                gr.update(value=None, visible=False))

    p = result["params"]
    active = INTENT_TILES.get(result["intent"], ())
    if args.cockpit_version == "v2":
        roll = p["roll"]
        bank = "wings level" if abs(roll) < 2 else ("right bank" if roll > 0 else "left bank")
        readings = tiles([
            ("altitude", "Altitude", f"{p['altitude']:,.0f} ft", ""),
            ("airspeed", "Airspeed", f"{max(p['airspeed'], 0):.0f} kt", ""),
            ("heading", "Heading", f"{p['heading']:03.0f}°", ""),
            ("pitch", "Pitch", f"{p['pitch']:+.1f}°", ""),
            ("roll", "Roll", f"{roll:+.1f}°", bank),
            ("vs", "Vertical speed", f"{p['vs']:+.0f} ft/min", ""),
            ("turn", "Turn rate", f"{p['turn']:+.2f}", "std-rate"),
            ("rpm", "Engine", f"{max(p['rpm'], 0):,.0f} RPM", ""),
            ("phase", "Flight phase", result["phase"].replace("_", " "),
             f"{result['phase_confidence'] * 100:.0f}% confidence"),
        ], active)
        sim = result["panel_similarity"]
        warnings = result.get("warnings") or []
        meta_items = [("Answered from", result["intent"].replace("_", " "), ""),
                      ("Routing", f"{result['confidence'] * 100:.0f}%", tone_for(result["confidence"])),
                      ("Time", fmt_ms(wall_ms), "")]
        if sim is not None:
            meta_items.append(("Panel match", f"{sim:.2f}", "good" if not warnings else "warn"))
        meta = chips(meta_items)
        if warnings:
            meta += "".join(f'<div class="vc-warn">⚠ {esc(w)}</div>' for w in warnings)
        details = kv_table([
            ("Model", system.model_info),
            ("Matched intent", result["intent"]),
            ("Routing confidence", f"{result['confidence'] * 100:.1f}% (question → intent router, "
                                   "not the gauge-reading model)"),
            ("Input", CROP_MODE_TEXT.get(result["crop_mode"], result["crop_mode"].replace("_", " "))),
            ("Panel similarity", "n/a" if sim is None else f"{sim:.3f}"),
        ] + timing_rows)
        return (result["answer"], meta, readings, details,
                gr.update(value=result["panel"], visible=result["panel"] is not None))

    # ---- v1 (kept for comparison only) ----
    readings = tiles([
        ("altitude", "Altitude", f"{p['altitude']:.0f} ft", ""),
        ("airspeed", "Airspeed", f"{p['airspeed']:.0f} kt", ""),
        ("heading", "Heading", f"{p['heading']:.0f}°", ""),
        ("pitch", "Pitch", f"{p['pitch_bin']}° bin", f"~{p['pitch_value']:.0f}°"),
        ("roll", "Roll", f"{p['roll_bin']}° bin", f"~{p['roll_value']:.0f}°"),
        ("phase", "Flight phase", result["phase"], "heuristic, not a classifier"),
    ], active)
    meta = chips([("Answered from", result["intent"].replace("_", " "), ""),
                  ("Routing", f"{result['confidence'] * 100:.0f}%", tone_for(result["confidence"])),
                  ("Time", fmt_ms(wall_ms), "")])
    details = kv_table([("Model", "CockpitNet (ResNet18), v1"), ("Matched intent", result["intent"])]
                       + timing_rows)
    return result["answer"], meta, readings, details, gr.update(value=None, visible=False)


# =========================================================================
# RADAR PANEL CALLBACKS (mirrors radar/app.py's answer_question)
# =========================================================================
RADAR_PRESET_QUESTIONS = [
    "How many aircraft are visible on the radar?",
    "Is there any aircraft on the left side of the radar?",
    "Is there any aircraft on the right side of the radar?",
    "What is the callsign of the aircraft with the highest altitude?",
    "Which quadrant of the radar has the most aircraft?",
]
# What actually produced a radar answer, from radar_system's "verdict" (falls back to "kind").
# Kept honest on purpose: answers read from rendered_frames.json are labelled as metadata, not as
# something the model read from the image.
RADAR_SOURCE_BY_VERDICT = {
    "yolo_first": "YOLO detector / frame metadata",
    "yolo_quadrant_count": "YOLO detector",
    "yolo_count": "YOLO detector",
    "yolo_metadata_match": "YOLO + frame metadata",
    "metadata_altitude_extreme": "frame metadata (not read from the image)",
    "confident": "radar CNN", "low_confidence_hedge": "radar CNN",
    "low_confidence_decline": "radar CNN", "yolo_mismatch": "radar CNN (YOLO disagrees)",
}
RADAR_SOURCE_BY_KIND = {"vqa": "radar CNN", "locate": "callsign lookup (frame metadata)",
                        "irrelevant": "out of scope"}


def radar_source(kind, verdict):
    if kind == "irrelevant":
        return RADAR_SOURCE_BY_KIND["irrelevant"]
    return RADAR_SOURCE_BY_VERDICT.get(verdict) or RADAR_SOURCE_BY_KIND.get(kind) or (kind or "n/a")


def radar_load():
    system, err = get_radar_system()
    if err:
        return status_pill("error", err)
    return status_pill("ready", "Model ready")


def radar_run_analysis(image_path, question):
    system, err = get_radar_system()

    if err:
        return err, "", "", gr.update(value=None, visible=False)

    if not image_path:
        return "Please upload a radar image first.", "", "", gr.update(value=None, visible=False)

    if not question or not question.strip():
        return "Please enter or select a question.", "", "", gr.update(value=None, visible=False)

    t0 = time.perf_counter()
    result = system.handle_question(image_path, question)
    wall_ms = (time.perf_counter() - t0) * 1000

    # Extract results
    kind = result.get("kind")
    conf = result.get("confidence")
    verdict = result.get("verdict")
    confidence_type = result.get("confidence_type", "model")

    source = radar_source(kind, verdict)

    # Metadata chips
    meta_items = []

    if kind or verdict:
        meta_items.append((
            "Answered by",
            source,
            "bad" if kind == "irrelevant" else ""
        ))

    from_metadata = verdict in ("yolo_metadata_match", "metadata_altitude_extreme") or kind == "locate"
    confidence_label = ("YOLO detection confidence (avg)" if confidence_type == "detector"
                        else "CNN model confidence")
    if conf is not None:
        meta_items.append((confidence_label, f"{conf * 100:.1f}%", tone_for(conf)))
    elif from_metadata:
        meta_items.append(("Confidence", "N/A — read from frame data", ""))
    elif kind != "irrelevant":
        meta_items.append(("Confidence", "N/A", ""))

    meta_items.append(("Time", fmt_ms(wall_ms), ""))

    # Detailed results table
    detail_rows = [
        ("Answered by", source),
        ("Answer type", kind or "n/a"),
        (confidence_label if conf is not None else "Confidence",
         f"{conf * 100:.1f}%" if conf is not None
         else ("N/A — read from frame data" if from_metadata else "N/A")),
        ("Verdict", verdict or "n/a"),
        ("Total", fmt_ms(wall_ms))
    ]

    img = result.get("image")

    return (
        result["answer"],
        chips(meta_items),
        kv_table(detail_rows),
        gr.update(value=img, visible=img is not None)
    )


# =========================================================================
# VOICE (browser-side JS; written by Asritha in commit 0f3d7e0, now shared by both panels)
# =========================================================================
# Voice input: the browser's built-in speech recognition (Web Speech API, Chrome/Edge) writes the
# spoken question into the question box; nothing is answered until the user clicks Run Analysis.
SPEAK_JS_TEMPLATE = r"""
(current) => new Promise((resolve) => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
        alert('Voice input needs Chrome or Edge (speech recognition is not supported in this browser).');
        resolve(current); return;
    }
    const btn = document.querySelector('#__BTN_ID__');
    const label = btn ? btn.textContent : '';
    const done = (value) => { if (btn) { btn.textContent = label; } resolve(value); };
    const rec = new SR();
    rec.lang = 'en-US';
    rec.interimResults = false;
    rec.continuous = false;
    rec.maxAlternatives = 1;
    let heard = null;
    rec.onresult = (e) => { heard = e.results[0][0].transcript.trim(); };
    rec.onerror = (e) => {
        const msg = {
            'not-allowed': 'Microphone access was blocked. Allow the microphone for this page and try again.',
            'service-not-allowed': 'Microphone access was blocked. Allow the microphone for this page and try again.',
            'no-speech': 'I could not hear a question. Click Speak and try again.',
            'audio-capture': 'No microphone was found.',
            'network': 'Speech recognition needs an internet connection.'
        }[e.error] || ('Voice input error: ' + e.error);
        alert(msg);
    };
    rec.onend = () => { done(heard ? heard : current); };
    if (btn) { btn.textContent = '🎙 Listening... speak now'; }
    try { rec.start(); } catch (err) { alert('Voice input error: ' + err); done(current); }
})
"""

# Voice output: browser SpeechSynthesis (no cost, no extra Python dependency), fired whenever the
# answer text changes - typed, preset or spoken question alike.
TTS_JS = r"""
(text, enabled) => {
    if (!enabled || !text || !('speechSynthesis' in window)) { return []; }
    const spoken = text
        .replace(/[*_`#]/g, '')
        .replace(/ft\/min/g, 'feet per minute')
        .replace(/\bkt\b/g, 'knots')
        .replace(/\bft\b/g, 'feet')
        .replace(/°/g, ' degrees')
        .replace(/\bYOLOv5\b/g, 'YOLO five')
        .replace(/\+(\d)/g, 'plus $1')
        .replace(/(^|\s)-(\d)/g, '$1minus $2');
    window.speechSynthesis.cancel();   // don't queue answers back-to-back
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(spoken));
    return [];
}
"""
STOP_TTS_JS = "() => { if ('speechSynthesis' in window) { window.speechSynthesis.cancel(); } return []; }"

# Always render in dark mode (otherwise Gradio follows the viewer's OS light/dark setting, and the
# custom dark styling below would clash with light-mode inputs on a light-mode laptop/projector).
FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"


# =========================================================================
# STYLE
# =========================================================================
CUSTOM_CSS = """
:root { --vc-bg:#0b0f19; --vc-panel:#121826; --vc-panel2:#161d2e; --vc-border:#232b3d;
        --vc-accent:#7c9cff; --vc-text:#e6e9f2; --vc-muted:#8891a5;
        --vc-good:#4ade80; --vc-warn:#fbbf24; --vc-bad:#f87171; }
.gradio-container { background-color: var(--vc-bg) !important; max-width: 1400px !important; }
body { background-color: var(--vc-bg) !important; }

/* header */
.vc-header { display:flex; align-items:center; gap:14px; padding:6px 2px 2px; }
.vc-title { font-size:1.5em; font-weight:700; color:var(--vc-text); line-height:1.1; }
.vc-subtitle { color:var(--vc-muted); font-size:0.92em; }

/* landing */
.vc-hero { padding:28px 4px 8px; }
.vc-hero h1 { font-size:2.1em; font-weight:750; color:var(--vc-text); margin:0 0 8px; }
.vc-hero p { color:var(--vc-muted); font-size:1.05em; max-width:760px; margin:0; }
.vc-steps { display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }
.vc-step { background:var(--vc-panel); border:1px solid var(--vc-border); border-radius:999px;
           padding:6px 14px; color:var(--vc-text); font-size:0.9em; }
.vc-step b { color:var(--vc-accent); margin-right:6px; }
.vc-card { background:var(--vc-panel) !important; border:1px solid var(--vc-border) !important;
           border-radius:16px !important; padding:22px !important; }
.vc-card:hover { border-color:#3a4766 !important; }
.vc-card-icon { width:84px; height:84px; margin-bottom:12px; }
.vc-card-title { font-size:1.35em; font-weight:700; color:var(--vc-text); margin-bottom:4px; }
.vc-card-desc { color:var(--vc-muted); margin-bottom:12px; }
.vc-card-btn { flex:0 0 auto !important; margin-top:auto !important; }
.vc-card-meta { color:var(--vc-muted); font-size:0.85em; margin-top:12px; }
.vc-footer { color:var(--vc-muted); font-size:0.85em; text-align:center; padding:26px 0 6px;
             border-top:1px solid var(--vc-border); margin-top:28px; }

/* chips + status */
.vc-chips { display:flex; flex-wrap:wrap; gap:8px; }
.vc-chip { background:var(--vc-panel2); border:1px solid var(--vc-border); border-radius:999px;
           padding:4px 12px; font-size:0.85em; color:var(--vc-text); }
.vc-chip-label { color:var(--vc-muted); margin-right:6px; }
.vc-chip.good { border-color:rgba(74,222,128,.45); } .vc-chip.warn { border-color:rgba(251,191,36,.5); }
.vc-chip.bad { border-color:rgba(248,113,113,.5); }
.vc-status { display:inline-flex; align-items:center; gap:8px; border-radius:999px; padding:5px 12px;
             font-size:0.85em; border:1px solid var(--vc-border); color:var(--vc-text);
             background:var(--vc-panel); }
.vc-dot { width:8px; height:8px; border-radius:50%; background:var(--vc-muted); }
.vc-status-loading .vc-dot { background:var(--vc-warn); animation:vcpulse 1s infinite; }
.vc-status-ready .vc-dot { background:var(--vc-good); }
.vc-status-error { border-color:rgba(248,113,113,.5); } .vc-status-error .vc-dot { background:var(--vc-bad); }
@keyframes vcpulse { 50% { opacity:.3; } }
.vc-warn { color:var(--vc-warn); font-size:0.88em; margin-top:8px; }

/* panels */
.vc-panelbar { align-items:center !important; }
.vc-module-title { font-size:1.25em; font-weight:700; color:var(--vc-text); }
.vc-module-sub { color:var(--vc-muted); font-size:0.88em; }
.vc-section { color:var(--vc-muted); font-size:0.78em; font-weight:600; letter-spacing:.08em;
              text-transform:uppercase; margin:4px 0 -4px; }
#cockpit_answer, #radar_answer { background:var(--vc-panel); border:1px solid var(--vc-border);
              border-left:4px solid var(--vc-accent); border-radius:12px; padding:18px 20px;
              min-height:72px; }
#cockpit_answer p, #radar_answer p { font-size:1.3em; line-height:1.45; color:var(--vc-text); margin:0; }

/* readings */
.vc-tiles { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
.vc-tile { background:var(--vc-panel); border:1px solid var(--vc-border); border-radius:12px;
           padding:12px 14px; }
.vc-tile.active { border-color:var(--vc-accent); box-shadow:0 0 0 1px var(--vc-accent) inset;
                  background:#16203a; }
.vc-tile-label { color:var(--vc-muted); font-size:0.78em; text-transform:uppercase; letter-spacing:.05em; }
.vc-tile-value { color:var(--vc-text); font-size:1.3em; font-weight:650; margin-top:2px;
                 font-variant-numeric:tabular-nums; }
.vc-tile-sub { color:var(--vc-muted); font-size:0.8em; }
.vc-empty { color:var(--vc-muted); border:1px dashed var(--vc-border); border-radius:12px; padding:18px;
            text-align:center; }
.vc-kv { display:grid; grid-template-columns:minmax(140px,35%) 1fr; font-size:0.9em; }
.vc-kv-k, .vc-kv-v { padding:7px 8px; border-bottom:1px solid var(--vc-border); }
.vc-kv-k { color:var(--vc-muted); }
.vc-kv-v { color:var(--vc-text); word-break:break-word; }
@media (max-width: 800px) { .vc-tiles { grid-template-columns:repeat(2,minmax(0,1fr)); } }
"""

HEADER_HTML = """
<div class="vc-header">
  <div><div class="vc-title">Visual Co-Pilot</div>
  <div class="vc-subtitle">Visual question answering for aircraft instruments and ATC radar</div></div>
</div>
"""

# Simple original icons (inline SVG): an attitude indicator and a radar scope.
COCKPIT_ICON = """
<svg class="vc-card-icon" viewBox="0 0 84 84" xmlns="http://www.w3.org/2000/svg">
  <defs><clipPath id="vcai"><circle cx="42" cy="42" r="30"/></clipPath></defs>
  <circle cx="42" cy="42" r="38" fill="#1b2335" stroke="#3a4766" stroke-width="2"/>
  <g clip-path="url(#vcai)" transform="rotate(-12 42 42)">
    <rect x="0" y="0" width="84" height="44" fill="#3b6fd8"/>
    <rect x="0" y="44" width="84" height="40" fill="#8a5a2b"/>
    <line x1="0" y1="44" x2="84" y2="44" stroke="#fff" stroke-width="1.5"/>
  </g>
  <path d="M24 44 h12 l6 5 l6 -5 h12" fill="none" stroke="#fbbf24" stroke-width="3" stroke-linejoin="round"/>
  <circle cx="42" cy="42" r="30" fill="none" stroke="#0b0f19" stroke-width="2"/>
</svg>
"""
RADAR_ICON = """
<svg class="vc-card-icon" viewBox="0 0 84 84" xmlns="http://www.w3.org/2000/svg">
  <circle cx="42" cy="42" r="38" fill="#0c2414" stroke="#2f6b3f" stroke-width="2"/>
  <circle cx="42" cy="42" r="26" fill="none" stroke="#2f6b3f"/>
  <circle cx="42" cy="42" r="13" fill="none" stroke="#2f6b3f"/>
  <line x1="4" y1="42" x2="80" y2="42" stroke="#2f6b3f"/><line x1="42" y1="4" x2="42" y2="80" stroke="#2f6b3f"/>
  <path d="M42 42 L42 4 A38 38 0 0 1 74 22 Z" fill="#4ade80" opacity="0.25"/>
  <circle cx="58" cy="24" r="3.5" fill="#4ade80"/><circle cx="26" cy="56" r="3.5" fill="#4ade80"/>
  <circle cx="60" cy="58" r="3.5" fill="#4ade80"/>
</svg>
"""


def card_html(icon, title, desc, capabilities, input_note):
    caps = "".join(f'<span class="vc-chip">{esc(c)}</span>' for c in capabilities)
    return (f'{icon}<div class="vc-card-title">{esc(title)}</div>'
            f'<div class="vc-card-desc">{esc(desc)}</div><div class="vc-chips">{caps}</div>'
            f'<div class="vc-card-meta">{esc(input_note)}</div>')


COCKPIT_CAPS = (["Altitude", "Airspeed", "Heading", "Bank / pitch", "Vertical speed", "Engine RPM",
                 "Flight phase"] if args.cockpit_version == "v2" else
                ["Altitude", "Airspeed", "Heading", "Banking", "Climbing", "Flight phase"])

THEME = gr.themes.Base(primary_hue="blue")


# =========================================================================
# GUI -- one landing view + two panels, toggled by visibility (no
# subprocess, no new server, no port/timing guesswork).
# =========================================================================
with gr.Blocks(title="Visual Co-Pilot") as demo:
    gr.HTML(HEADER_HTML)

    # ---------------- Landing view ----------------
    with gr.Column(visible=True) as landing_view:
        gr.HTML("""
        <div class="vc-hero">
          <h1>Ask questions about what the aircraft's screens show.</h1>
          <p>Upload a cockpit instrument panel or a radar display, ask a question in plain English
             (typed or spoken), and get an answer read from the image.</p>
          <div class="vc-steps">
            <span class="vc-step"><b>1</b>Choose a module</span>
            <span class="vc-step"><b>2</b>Upload an image</span>
            <span class="vc-step"><b>3</b>Ask a question</span>
            <span class="vc-step"><b>4</b>Get the answer</span>
          </div>
        </div>""")
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="vc-card"):
                gr.HTML(card_html(COCKPIT_ICON, "Cockpit instruments",
                                  "Reads the Cessna 172 instrument panel and answers questions about "
                                  "the aircraft's state.", COCKPIT_CAPS,
                                  "Input: FlightGear screenshot or instrument-panel crop"))
                cockpit_card_btn = gr.Button("Open Cockpit module", variant="primary", size="lg",
                                             elem_classes="vc-card-btn")
            with gr.Column(elem_classes="vc-card"):
                gr.HTML(card_html(RADAR_ICON, "ATC radar",
                                  "Detects aircraft on a radar display and answers questions about "
                                  "traffic.", ["Aircraft count", "Position / quadrant", "Altitude",
                                               "Callsign"],
                                  "Input: radar display image"))
                radar_card_btn = gr.Button("Open Radar module", variant="primary", size="lg",
                                           elem_classes="vc-card-btn")
        gr.HTML('<div class="vc-footer">PES University capstone project · Krupa R · Madhumitha N V · '
                'Krithika A · M Asritha · Guide: Dr Vandana M L</div>')

    # ---------------- Cockpit panel ----------------
    with gr.Column(visible=False) as cockpit_view:
        with gr.Row(elem_classes="vc-panelbar"):
            cockpit_back_btn = gr.Button("← Modules", size="sm", scale=0, min_width=120)
            gr.HTML('<div class="vc-module-title">Cockpit instruments</div>'
                    '<div class="vc-module-sub">Cessna 172 panel · FlightGear</div>')
            cockpit_status = gr.HTML(status_pill("idle", "Model not loaded"))
        with gr.Row():
            with gr.Column(scale=5):
                gr.HTML('<div class="vc-section">Input</div>')
                cockpit_image_input = gr.Image(type="pil", label="Cockpit image (full FlightGear "
                                               "screenshot or instrument-panel crop)",
                                               sources=["upload", "clipboard"], height=380)
                cockpit_preset_dropdown = gr.Dropdown(
                    choices=COCKPIT_PRESET_QUESTIONS, label="Sample questions", value=None)
                cockpit_question_box = gr.Textbox(
                    label="Question", placeholder="Ask a question about the cockpit instruments...",
                    lines=2)
                with gr.Row():
                    cockpit_speak_btn = gr.Button("🎤 Speak", elem_id="cockpit_speak_btn")
                    cockpit_redo_btn = gr.Button("🔄 Clear question")
                cockpit_run_btn = gr.Button("Run Analysis", variant="primary", size="lg")
            with gr.Column(scale=6):
                gr.HTML('<div class="vc-section">Answer</div>')
                cockpit_answer_output = gr.Markdown(
                    "Upload an image and ask a question to get started.", elem_id="cockpit_answer")
                cockpit_meta_output = gr.HTML()
                with gr.Row():
                    cockpit_tts_checkbox = gr.Checkbox(label="🔊 Read answer aloud", value=True)
                    cockpit_stop_speech_btn = gr.Button("⏹ Stop", size="sm", scale=0, min_width=90)
                gr.HTML('<div class="vc-section">Instrument readings</div>')
                cockpit_readings_output = gr.HTML(EMPTY_READINGS)
                with gr.Accordion("Details (model, timing, what the model read)", open=False):
                    cockpit_details_output = gr.HTML()
                    cockpit_panel_output = gr.Image(label="What the model read (instrument panel)",
                                                    interactive=False, visible=False)

        cockpit_preset_dropdown.change(fn=lambda q: q, inputs=cockpit_preset_dropdown,
                                        outputs=cockpit_question_box)
        cockpit_run_btn.click(
            fn=cockpit_run_analysis,
            inputs=[cockpit_image_input, cockpit_question_box],
            outputs=[cockpit_answer_output, cockpit_meta_output, cockpit_readings_output,
                     cockpit_details_output, cockpit_panel_output],
        )
        cockpit_speak_btn.click(fn=None, inputs=[cockpit_question_box], outputs=[cockpit_question_box],
                                js=SPEAK_JS_TEMPLATE.replace("__BTN_ID__", "cockpit_speak_btn"))
        cockpit_redo_btn.click(fn=cockpit_redo_voice, outputs=[cockpit_question_box])
        cockpit_answer_output.change(fn=None, inputs=[cockpit_answer_output, cockpit_tts_checkbox],
                                     outputs=None, js=TTS_JS)
        cockpit_stop_speech_btn.click(fn=None, inputs=None, outputs=None, js=STOP_TTS_JS)

    # ---------------- Radar panel ----------------
    with gr.Column(visible=False) as radar_view:
        with gr.Row(elem_classes="vc-panelbar"):
            radar_back_btn = gr.Button("← Modules", size="sm", scale=0, min_width=120)
            gr.HTML('<div class="vc-module-title">ATC radar</div>'
                    '<div class="vc-module-sub">Aircraft detection + radar VQA</div>')
            radar_status = gr.HTML(status_pill("idle", "Model not loaded"))
        with gr.Row():
            with gr.Column(scale=5):
                gr.HTML('<div class="vc-section">Input</div>')
                radar_image_input = gr.Image(type="filepath", label="Radar image",
                                             sources=["upload", "clipboard"], height=380)
                radar_preset_dropdown = gr.Dropdown(
                    choices=RADAR_PRESET_QUESTIONS, label="Sample questions", value=None)
                radar_question_box = gr.Textbox(
                    label="Question", placeholder="Ask a question about the aircraft on the radar...",
                    lines=2)
                with gr.Row():
                    radar_speak_btn = gr.Button("🎤 Speak", elem_id="radar_speak_btn")
                    radar_redo_btn = gr.Button("🔄 Clear question")
                radar_run_btn = gr.Button("Run Analysis", variant="primary", size="lg")
            with gr.Column(scale=6):
                gr.HTML('<div class="vc-section">Answer</div>')
                radar_answer_output = gr.Markdown(
                    "Upload an image and ask a question to get started.", elem_id="radar_answer")
                radar_meta_output = gr.HTML()
                with gr.Row():
                    radar_tts_checkbox = gr.Checkbox(label="🔊 Read answer aloud", value=True)
                    radar_stop_speech_btn = gr.Button("⏹ Stop", size="sm", scale=0, min_width=90)
                radar_highlight_output = gr.Image(label="Highlighted aircraft", interactive=False,
                                                  visible=False)
                with gr.Accordion("Details", open=False):
                    radar_details_output = gr.HTML()

        radar_preset_dropdown.change(fn=lambda q: q, inputs=radar_preset_dropdown,
                                      outputs=radar_question_box)
        radar_run_btn.click(
            fn=radar_run_analysis,
            inputs=[radar_image_input, radar_question_box],
            outputs=[radar_answer_output, radar_meta_output, radar_details_output,
                     radar_highlight_output],
        )
        radar_speak_btn.click(fn=None, inputs=[radar_question_box], outputs=[radar_question_box],
                              js=SPEAK_JS_TEMPLATE.replace("__BTN_ID__", "radar_speak_btn"))
        radar_redo_btn.click(fn=lambda: "", outputs=[radar_question_box], js=STOP_TTS_JS)
        radar_answer_output.change(fn=None, inputs=[radar_answer_output, radar_tts_checkbox],
                                   outputs=None, js=TTS_JS)
        radar_stop_speech_btn.click(fn=None, inputs=None, outputs=None, js=STOP_TTS_JS)

    # ---------------- View switching (landing <-> cockpit <-> radar) ----------------
    VIEWS = [landing_view, cockpit_view, radar_view]

    def show_cockpit():
        loaded = _cockpit_system is not None
        return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                status_pill("ready", "Model ready") if loaded else
                status_pill("loading", "Loading model… (first time only)"))

    def show_radar():
        loaded = _radar_system is not None
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                status_pill("ready", "Model ready") if loaded else
                status_pill("loading", "Loading model… (first time only)"))

    def show_landing():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    # Switch view first, then load the model (so the panel appears immediately with a status pill).
    cockpit_card_btn.click(fn=show_cockpit, outputs=VIEWS + [cockpit_status]).then(
        fn=cockpit_load, outputs=[cockpit_status])
    radar_card_btn.click(fn=show_radar, outputs=VIEWS + [radar_status]).then(
        fn=radar_load, outputs=[radar_status])
    cockpit_back_btn.click(fn=show_landing, outputs=VIEWS)
    radar_back_btn.click(fn=show_landing, outputs=VIEWS)

if __name__ == "__main__":
    demo.launch(share=args.share, theme=THEME, css=CUSTOM_CSS, js=FORCE_DARK_JS,
                footer_links=["gradio", "settings"])