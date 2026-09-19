r"""
app.py
Gradio UI for the Visual Co-Pilot VQA system.

Setup:
  pip install gradio torch torchvision pandas pillow sentence-transformers faster-whisper --break-system-packages

Run:
  python app.py --data_dir "C:\Users\msasr\OneDrive\Desktop\cap_data\consolidated"

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import argparse
import gradio as gr
from PIL import Image

from copilot_core import CopilotSystem

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--img_width", type=int, default=224)
parser.add_argument("--img_height", type=int, default=224)
parser.add_argument("--use_letterbox", action="store_true")
parser.add_argument("--confidence_threshold", type=float, default=0.35)
parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
args = parser.parse_args()

print("Loading Visual Co-Pilot model...")
system = CopilotSystem(
    data_dir=args.data_dir,
    img_width=args.img_width,
    img_height=args.img_height,
    use_letterbox=args.use_letterbox,
    confidence_threshold=args.confidence_threshold,
)
print(f"Loaded checkpoint from epoch {system.checkpoint_epoch}")
if not system.have_embedder:
    print("WARNING: sentence-transformers not installed — question routing will use "
          "simple keyword matching instead of semantic similarity.")

# Lazy-loaded speech-to-text — only imported if the user actually uses voice input,
# so the app still starts fine without faster-whisper installed.
_whisper_model = None


def transcribe_audio(audio_path):
    global _whisper_model
    if audio_path is None:
        return ""
    try:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        segments, _ = _whisper_model.transcribe(audio_path)
        text = " ".join(seg.text for seg in segments).strip()
        return text
    except ImportError:
        return "[Voice input unavailable — install faster-whisper: pip install faster-whisper]"
    except Exception as e:
        return f"[Transcription error: {e}]"


PRESET_QUESTIONS = [
    "What is the current altitude?",
    "How fast are we going?",
    "What heading are we on?",
    "Are we banking left or right?",
    "Are we climbing?",
    "What phase of flight are we in?",
]


def run_analysis(image, question):
    if image is None:
        return "Please upload a cockpit panel image first.", "", "", 0.0
    if not question or not question.strip():
        return "Please enter or select a question.", "", "", 0.0

    result = system.answer(image, question)
    p = result["params"]

    key_detections_md = (
        f"- **Altitude:** {p['altitude']:.0f} ft\n"
        f"- **Airspeed:** {p['airspeed']:.0f} kt\n"
        f"- **Heading:** {p['heading']:.0f}°\n"
        f"- **Pitch:** {p['pitch_bin']}° bin (~{p['pitch_value']:.0f}°)\n"
        f"- **Roll:** {p['roll_bin']}° bin (~{p['roll_value']:.0f}°)\n"
        f"- **Inferred phase:** {result['phase']} _(heuristic, not a certified classifier)_"
    )

    footer_md = (
        f"**Model:** CockpitNet (ResNet18)  •  "
        f"**Latency:** {result['timings']['total_ms']:.0f}ms  •  "
        f"**Matched intent:** {result['intent']}"
    )

    confidence_pct = result["confidence"] * 100
    return result["answer"], key_detections_md, footer_md, confidence_pct


CUSTOM_CSS = """
.gradio-container {background-color: #0b0f19 !important;}
#title {color: #7c9cff; font-size: 1.6em; font-weight: 700; margin-bottom: 0.2em;}
#answer_box {background-color: #131824; border: 1px solid #232b3d; border-radius: 10px; padding: 16px;}
#detections_box {background-color: #131824; border: 1px solid #232b3d; border-radius: 10px; padding: 16px;}
#footer_box {color: #8891a5; font-size: 0.85em; margin-top: 8px;}
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(primary_hue="blue")) as demo:
    gr.Markdown("### ✈️ Visual Co-Pilot — Aviation VQA", elem_id="title")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload cockpit panel image",
                                    sources=["upload", "clipboard"])

            preset_dropdown = gr.Dropdown(
                choices=PRESET_QUESTIONS, label="Sample questions", value=None)

            question_box = gr.Textbox(
                label="Question", placeholder="Ask a question about the cockpit instruments...",
                lines=2)

            with gr.Row():
                mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Or ask by voice")

            run_btn = gr.Button("Run Analysis", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("**Analysis Result**")
            answer_output = gr.Markdown(elem_id="answer_box")
            confidence_output = gr.Slider(
                label="Routing confidence", minimum=0, maximum=100, value=0, interactive=False)
            gr.Markdown("**Key Detections**")
            detections_output = gr.Markdown(elem_id="detections_box")
            footer_output = gr.Markdown(elem_id="footer_box")

    # Preset dropdown fills the question box (still editable afterward)
    preset_dropdown.change(fn=lambda q: q, inputs=preset_dropdown, outputs=question_box)

    # Voice input transcribes into the question box (still editable afterward)
    mic_input.change(fn=transcribe_audio, inputs=mic_input, outputs=question_box)

    run_btn.click(
        fn=run_analysis,
        inputs=[image_input, question_box],
        outputs=[answer_output, detections_output, footer_output, confidence_output],
    )

if __name__ == "__main__":
    demo.launch(share=args.share)
