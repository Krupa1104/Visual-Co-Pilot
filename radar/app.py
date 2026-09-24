r"""
radar/app.py
Gradio UI for the Radar VQA module -- mirrors cockpit/gui/app.py's pattern
so both modules have a real, standalone, launchable GUI (needed for
integrate.py to route to either one symmetrically).

Setup:
  pip install gradio torch torchvision pillow --break-system-packages
  (plus whatever radar/requirements-radar.txt lists, e.g. YOLOv5's deps)

Run FROM INSIDE radar/ (same convention as radar_cli.py), so
`from radar_system import RadarSystem` resolves:
  cd radar
  python app.py --base_dir "C:\...\radar_base" --yolo_root "C:\...\yolov5"

Then open the local URL Gradio prints (default http://127.0.0.1:7860).
Use --server_port to run alongside cockpit's app.py at the same time
without a port clash (cockpit defaults to Gradio's own default port too).
"""

import argparse
import gradio as gr
from PIL import Image

from radar_system import RadarSystem

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", required=True,
                     help="Local folder holding models/, data/, logs/ "
                          "(your local copy of aviation_vqa_output from Drive)")
parser.add_argument("--repo_root", default=None,
                     help="Path to your Radar_Phase1 checkout root. Default: current directory.")
parser.add_argument("--yolo_root", default=None,
                     help="Path to a local yolov5 checkout. Default: <parent of base_dir>/yolov5")
parser.add_argument("--yolo_conf", type=float, default=0.25)
parser.add_argument("--cnn_checkpoint", default="cnn_best_model.pth",
                     help="Filename inside <base_dir>/models/ to load. Try "
                          "cnn_best_model.pth, best_model.pth, or final_model.pth.")
parser.add_argument("--server_port", type=int, default=None,
                     help="Gradio port. Leave unset to let Gradio pick its default "
                          "(useful if you're also running cockpit's app.py at the same time).")
parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
args = parser.parse_args()

print(f"Loading RadarSystem (checkpoint: {args.cnn_checkpoint})...")
system = RadarSystem(
    base_dir=args.base_dir,
    repo_root=args.repo_root,
    yolo_root=args.yolo_root,
    yolo_conf=args.yolo_conf,
    cnn_checkpoint=args.cnn_checkpoint,
)
print("RadarSystem loaded.")


def answer_question(image: Image.Image, question: str, saved_image_path: str):
    if image is None:
        return "Please upload or select a radar image first.", None
    if not question or not question.strip():
        return "Please enter a question.", None
    if not saved_image_path:
        return ("Internal error: could not resolve a file path for the uploaded "
                "image -- handle_question needs a path, not just pixel data. "
                "Try re-uploading the image."), None

    result = system.handle_question(saved_image_path, question)
    return result["answer"], result.get("image")


def on_image_upload(image_path):
    # Gradio's gr.Image with type="filepath" already gives us a real path on
    # disk (either the upload's temp copy or the original path if selecting
    # from a local example) -- RadarSystem.handle_question needs a path, not
    # raw pixels, so we keep it in a hidden state box rather than passing the
    # PIL image through and losing the path.
    return image_path


with gr.Blocks(title="Radar VQA -- Visual Co-Pilot") as demo:
    gr.Markdown("## Radar VQA\nUpload a radar frame and ask a question about the aircraft on it.")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Radar image", type="filepath")
            question_input = gr.Textbox(
                label="Question",
                placeholder="e.g. How many aircraft are visible on the radar?",
            )
            ask_button = gr.Button("Ask", variant="primary")
            gr.Examples(
                examples=[[q] for q in [
                    "How many aircraft are visible on the radar?",
                    "Is there any aircraft on the left side of the radar?",
                    "What is the callsign of the aircraft with the highest altitude?",
                    "Which quadrant of the radar has the most aircraft?",
                ]],
                inputs=[question_input],
                label="Example questions",
            )
        with gr.Column():
            answer_output = gr.Textbox(label="Answer", interactive=False)
            highlight_output = gr.Image(label="Highlighted result (locate/find queries only)")

    image_path_state = gr.State(None)
    image_input.change(on_image_upload, inputs=[image_input], outputs=[image_path_state])
    ask_button.click(
        answer_question,
        inputs=[image_input, question_input, image_path_state],
        outputs=[answer_output, highlight_output],
    )
    question_input.submit(
        answer_question,
        inputs=[image_input, question_input, image_path_state],
        outputs=[answer_output, highlight_output],
    )

if __name__ == "__main__":
    demo.launch(server_port=args.server_port, share=args.share)