r"""
radar_cli.py
Run this FROM INSIDE your Radar_Phase1 checkout (or pass --repo_root),
so `from models.cnn_inference import ...` and `from radar_generation...`
resolve to your real code.

Usage:
  cd Visual-Co-Pilot
  python radar_cli.py --base_dir "radar_base" --yolo_root "yolov5" \
      --image "radar_base\data\raw\images\frame_0007.png" \
      --question "how many aircraft are visible on the radar?" \
      --cnn_checkpoint "best_model.pth"
"""

import os
import argparse

from radar_system import RadarSystem

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", required=True,
                     help="Local folder holding models/, data/, logs/ "
                          "(your local copy of aviation_vqa_output from Drive)")
parser.add_argument("--repo_root", default=None,
                     help="Path to your Radar_Phase1 checkout root (contains models/, "
                          "radar_generation/ as folders). Default: current directory.")
parser.add_argument("--yolo_root", default=None,
                     help="Path to a local `git clone https://github.com/ultralytics/yolov5` "
                          "checkout. Default: <parent of base_dir>/yolov5")
parser.add_argument("--image", required=True)
parser.add_argument("--question", default=None)
parser.add_argument("--yolo_conf", type=float, default=0.25)
parser.add_argument("--cnn_checkpoint", default="cnn_best_model.pth",
                     help="Filename (inside <base_dir>/models/) of the CNN checkpoint to load. "
                          "Try cnn_best_model.pth, best_model.pth, or final_model.pth -- "
                          "whichever one actually matches the current models/cnn_model.py architecture.")
args = parser.parse_args()

print(f"Loading RadarSystem (checkpoint: {args.cnn_checkpoint})...")
system = RadarSystem(
    base_dir=args.base_dir,
    repo_root=args.repo_root,
    yolo_root=args.yolo_root,
    yolo_conf=args.yolo_conf,
    cnn_checkpoint=args.cnn_checkpoint,
)
print("RadarSystem ready.")
print("LLM relevance gate: ON (Qwen2.5-1.5B-Instruct)" if system.have_llm_gate
      else "LLM relevance gate: OFF -- using keyword fallback")


def run_and_print(question):
    result = system.handle_question(args.image, question)
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"  kind={result['kind']} confidence={result['confidence']} verdict={result['verdict']}")
    if result.get("image") is not None:
        stem, ext = os.path.splitext(args.image)
        out_path = f"{stem}_highlighted.png"
        result["image"].save(out_path)
        print(f"  annotated image saved to: {out_path}")


if args.question:
    run_and_print(args.question)
else:
    print("\nEnter questions (blank line to quit):")
    while True:
        q = input("Q: ").strip()
        if not q:
            break
        run_and_print(q)