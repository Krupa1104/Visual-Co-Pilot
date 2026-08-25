r"""
vqa_system.py
CLI interface for the Visual Co-Pilot VQA system. All actual logic lives
in copilot_core.py (shared with app.py, the Gradio UI) — this file is
just argument parsing and printing.

Usage (single question):
  python vqa_system.py --data_dir "C:\...\consolidated" --image "climb_0005.png" --question "how high are we flying"

Usage (interactive loop):
  python vqa_system.py --data_dir "C:\...\consolidated" --image "climb_0005.png"
"""

import argparse
from copilot_core import CopilotSystem

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--question", default=None)
parser.add_argument("--img_width", type=int, default=224)
parser.add_argument("--img_height", type=int, default=224)
parser.add_argument("--use_letterbox", action="store_true",
                     help="Only set this if best_model.pt was trained with letterbox "
                          "preprocessing. Default (off) matches the original epoch-16 checkpoint.")
parser.add_argument("--confidence_threshold", type=float, default=0.35)
args = parser.parse_args()

print("Loading model...")
system = CopilotSystem(
    data_dir=args.data_dir,
    img_width=args.img_width,
    img_height=args.img_height,
    use_letterbox=args.use_letterbox,
    confidence_threshold=args.confidence_threshold,
)
print(f"Loaded checkpoint from epoch {system.checkpoint_epoch}")
print("Using LETTERBOX preprocessing." if args.use_letterbox else
      "Using PLAIN (stretch) resize — matches the original epoch-16 checkpoint.")
if not system.have_embedder:
    print("WARNING: sentence-transformers not installed — falling back to simple keyword matching.")


def print_result(result):
    p = result["params"]
    print(f"\nPredicted parameters:")
    print(f"  altitude: {p['altitude']:.0f} ft")
    print(f"  airspeed: {p['airspeed']:.0f} kt")
    print(f"  heading: {p['heading']:.0f} deg")
    print(f"  pitch_bin: {p['pitch_bin']} (~{p['pitch_value']:.0f} deg)")
    print(f"  roll_bin: {p['roll_bin']} (~{p['roll_value']:.0f} deg)")
    print(f"  inferred phase (heuristic, not a trained classifier): {result['phase']}")
    print(f"  latency: {result['timings']['total_ms']:.0f}ms "
          f"(preprocess={result['timings']['preprocess_ms']:.0f}ms, "
          f"inference={result['timings']['inference_ms']:.0f}ms, "
          f"routing={result['timings']['routing_ms']:.0f}ms)")


print(f"\nRunning perception on: {args.image}")

if args.question:
    result = system.answer(args.image, args.question)
    print_result(result)
    print(f"\nQ: {args.question}")
    print(f"A: {result['answer']}")
    print(f"  (matched intent: {result['intent']}, confidence: {result['confidence']:.2f})")
else:
    # Run perception once, then allow many questions against the same image
    params, _ = system.run_perception(args.image)
    phase = system.infer_phase(params)
    print(f"\nPredicted parameters:")
    print(f"  altitude: {params['altitude']:.0f} ft")
    print(f"  airspeed: {params['airspeed']:.0f} kt")
    print(f"  heading: {params['heading']:.0f} deg")
    print(f"  pitch_bin: {params['pitch_bin']} (~{params['pitch_value']:.0f} deg)")
    print(f"  roll_bin: {params['roll_bin']} (~{params['roll_value']:.0f} deg)")
    print(f"  inferred phase (heuristic, not a trained classifier): {phase}")

    print("\nEnter questions (blank line to quit):")
    while True:
        q = input("Q: ").strip()
        if not q:
            break
        intent, score, routing_ms = system.route_question(q)
        answer_text = system.generate_answer(intent, params, phase, score)
        print(f"A: {answer_text}")
        print(f"  (matched intent: {intent}, confidence: {score:.2f}, routing: {routing_ms:.0f}ms)\n")
