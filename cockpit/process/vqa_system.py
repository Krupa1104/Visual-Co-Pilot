r"""
vqa_system.py
Visual Co-Pilot VQA wrapper.

Combines:
  1. The trained CNN (perception): image -> altitude, airspeed, heading,
     pitch_bin, roll_bin
  2. A simple heuristic phase inference (context) from those parameters
  3. An intent router: natural-language question -> which parameter/intent
     is being asked about, via sentence-embedding similarity (handles
     paraphrasing, not just exact keyword matches)
  4. A template-based answer generator: parameter values + phase context
     -> natural language answer

Usage (single question):
  python vqa_system.py --data_dir "C:\...\consolidated" --image "climb_0005.png" --question "how high are we flying"

Usage (interactive loop):
  python vqa_system.py --data_dir "C:\...\consolidated" --image "climb_0005.png"
"""

import os
import json
import argparse

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="Folder containing model_output/ (from train_model.py)")
parser.add_argument("--image", required=True, help="Path to a cockpit panel image to answer questions about")
parser.add_argument("--question", default=None, help="A single question to ask")
parser.add_argument("--questions_file", default=None,
                     help="Path to a .txt file with one question per line — runs all of them against --image")
parser.add_argument("--img_width", type=int, default=224)
parser.add_argument("--img_height", type=int, default=224)
parser.add_argument("--confidence_threshold", type=float, default=0.35,
                     help="Below this similarity score, the router admits it doesn't understand the question")
parser.add_argument("--use_letterbox", action="store_true",
                     help="Use letterbox (pad, no distortion) preprocessing. ONLY set this "
                          "if best_model.pt was trained with the LetterboxResize transform. "
                          "The original checkpoint (epoch 16, the one validated by "
                          "eval_altitude_error.py) was trained with a plain non-uniform "
                          "resize, NOT letterbox — using the wrong one here causes a "
                          "train/inference mismatch that can badly degrade predictions.")
args = parser.parse_args()

MODEL_DIR = os.path.join(args.data_dir, "model_output")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# LOAD TRAINED MODEL
# =========================================================================
with open(os.path.join(MODEL_DIR, "norm_stats.json")) as f:
    norm_stats = json.load(f)
with open(os.path.join(MODEL_DIR, "label_maps.json")) as f:
    label_maps = json.load(f)

PITCH_LABELS = label_maps["pitch_labels"]
ROLL_LABELS = label_maps["roll_labels"]


class LetterboxResize:
    def __init__(self, target_width, target_height, fill=(0, 0, 0)):
        self.target_width = target_width
        self.target_height = target_height
        self.fill = fill

    def __call__(self, img):
        src_w, src_h = img.size
        scale = min(self.target_width / src_w, self.target_height / src_h)
        new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.target_width, self.target_height), self.fill)
        paste_x = (self.target_width - new_w) // 2
        paste_y = (self.target_height - new_h) // 2
        canvas.paste(img_resized, (paste_x, paste_y))
        return canvas


class CockpitNet(nn.Module):
    def __init__(self, num_pitch_classes, num_roll_classes):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=None)
        backbone_out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.altitude_head = nn.Linear(backbone_out_dim, 1)
        self.airspeed_head = nn.Linear(backbone_out_dim, 1)
        self.heading_head = nn.Linear(backbone_out_dim, 2)
        self.pitch_head = nn.Linear(backbone_out_dim, num_pitch_classes)
        self.roll_head = nn.Linear(backbone_out_dim, num_roll_classes)

    def forward(self, x):
        features = self.backbone(x)
        return {
            "altitude": self.altitude_head(features).squeeze(-1),
            "airspeed": self.airspeed_head(features).squeeze(-1),
            "heading_sincos": self.heading_head(features),
            "pitch_logits": self.pitch_head(features),
            "roll_logits": self.roll_head(features),
        }


model = CockpitNet(len(PITCH_LABELS), len(ROLL_LABELS)).to(DEVICE)
checkpoint = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

if args.use_letterbox:
    print("Using LETTERBOX preprocessing — make sure best_model.pt was trained with this too.")
    resize_step = LetterboxResize(args.img_width, args.img_height)
else:
    print("Using PLAIN (stretch) resize — matches the original epoch-16 checkpoint.")
    resize_step = transforms.Resize((args.img_height, args.img_width))

transform = transforms.Compose([
    resize_step,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def run_perception(image_path):
    """Image -> dict of predicted flight parameters."""
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_t)

    altitude = pred["altitude"].item() * norm_stats["altitude"]["std"] + norm_stats["altitude"]["mean"]
    airspeed = pred["airspeed"].item() * norm_stats["airspeed"]["std"] + norm_stats["airspeed"]["mean"]

    sin_h, cos_h = pred["heading_sincos"][0].tolist()
    import math
    heading = math.degrees(math.atan2(sin_h, cos_h)) % 360

    pitch_idx = pred["pitch_logits"].argmax(1).item()
    roll_idx = pred["roll_logits"].argmax(1).item()
    pitch_bin = PITCH_LABELS[pitch_idx]
    roll_bin = ROLL_LABELS[roll_idx]

    # Use the bin edges directly (not string parsing, which is fragile for
    # doubly-negative labels like "-3--1") to get a representative midpoint
    # value for the predicted pitch/roll bin.
    pitch_edges = label_maps["pitch_bin_edges"]
    roll_edges = label_maps["roll_bin_edges"]
    pitch_mid = (pitch_edges[pitch_idx] + pitch_edges[pitch_idx + 1]) / 2
    roll_mid = (roll_edges[roll_idx] + roll_edges[roll_idx + 1]) / 2

    return {
        "altitude": altitude,
        "airspeed": airspeed,
        "heading": heading,
        "pitch_bin": pitch_bin,
        "roll_bin": roll_bin,
        "pitch_value": pitch_mid,
        "roll_value": roll_mid,
    }


def infer_phase(params):
    """Heuristic phase context from predicted parameters — not a trained
    classifier, just rule-based bucketing matching how the dataset phases
    were originally defined. Used only to make answers context-aware
    (e.g. distinguishing 'taxiing' from 'climbing' for phrasing), not
    presented as a certified phase classification.

    Thresholds include buffer margin for observed model noise (~150-250ft
    altitude, ~5-10kt speed) rather than the exact dataset boundaries —
    tight boundaries with no margin cause narrow misses to fall through
    to the wrong bucket even when the underlying prediction is basically
    correct."""
    alt = params["altitude"]
    spd = params["airspeed"]
    pitch = params["pitch_value"]

    if alt < 150:
        # On the ground: use speed to distinguish parked / taxiing / accelerating
        if spd < 10:
            return "engine_off"
        elif spd < 40:
            return "taxi"
        else:
            return "takeoff_roll"
    elif alt < 500:
        # Could be finishing the takeoff roll or just starting to climb out —
        # use pitch to disambiguate, since altitude alone is ambiguous here
        if pitch >= 3:
            return "climb"
        else:
            return "takeoff_roll"
    elif pitch >= 3 and alt < 4500:
        return "climb"
    else:
        return "level_flight"


# =========================================================================
# INTENT ROUTER (sentence-embedding based, handles paraphrasing)
# =========================================================================
try:
    from sentence_transformers import SentenceTransformer, util
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    HAVE_EMBEDDER = True
except ImportError:
    print("WARNING: sentence-transformers not installed — falling back to "
          "simple keyword matching. Install with: pip install sentence-transformers")
    HAVE_EMBEDDER = False

INTENT_EXAMPLES = {
    "get_altitude": ["what is the altitude", "how high are we", "how high are we flying",
                      "current elevation", "altimeter reading", "what altitude are we at"],
    "get_airspeed": ["what is the airspeed", "how fast are we going", "current speed",
                       "what speed are we flying at", "airspeed indicator reading"],
    "get_heading": ["what heading are we on", "which direction are we flying",
                     "current compass heading", "what direction is the plane pointed"],
    "is_banking": ["are we banking", "is the plane turning", "are we tilted left or right",
                    "is the aircraft rolling", "how much are we banked"],
    "is_climbing": ["are we climbing", "are we gaining altitude", "is the nose up",
                      "are we ascending", "is the aircraft climbing"],
    "flight_phase": ["what phase of flight are we in", "what is the aircraft doing right now",
                       "are we taking off or landing", "what's our current status"],
}

if HAVE_EMBEDDER:
    intent_texts = []
    intent_labels = []
    for intent, examples in INTENT_EXAMPLES.items():
        for ex in examples:
            intent_texts.append(ex)
            intent_labels.append(intent)
    intent_embeddings = embedder.encode(intent_texts, convert_to_tensor=True)


def route_question(question):
    if not HAVE_EMBEDDER:
        # crude fallback: keyword substring match
        q = question.lower()
        for intent, examples in INTENT_EXAMPLES.items():
            for ex in examples:
                if any(word in q for word in ex.split()):
                    return intent, 1.0
        return None, 0.0

    q_embedding = embedder.encode(question, convert_to_tensor=True)
    similarities = util.cos_sim(q_embedding, intent_embeddings)[0]
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()
    best_intent = intent_labels[best_idx]
    return best_intent, best_score


# =========================================================================
# ANSWER GENERATOR
# =========================================================================
def generate_answer(intent, params, phase, score):
    if intent is None or score < args.confidence_threshold:
        return ("I'm not confident I understood that question. I can currently answer "
                "questions about altitude, airspeed, heading, banking/roll, climbing, "
                "and general flight phase.")

    if intent == "get_altitude":
        return f"The current altitude is approximately {params['altitude']:.0f} ft."

    if intent == "get_airspeed":
        return f"The current airspeed is approximately {params['airspeed']:.0f} kt."

    if intent == "get_heading":
        return f"The aircraft is currently heading approximately {params['heading']:.0f} degrees."

    if intent == "is_banking":
        roll = params["roll_value"]
        if abs(roll) < 2:
            return "The aircraft's wings are approximately level — no significant bank."
        direction = "left" if roll < 0 else "right"
        return f"Yes, the aircraft is in a {direction} bank of roughly {abs(roll):.0f} degrees."

    if intent == "is_climbing":
        if phase == "climb":
            return f"Yes, the aircraft is climbing, with a pitch of roughly {params['pitch_value']:.0f} degrees nose-up."
        if phase in ("engine_off", "taxi"):
            return "No, the aircraft is currently on the ground, not climbing."
        if phase == "takeoff_roll":
            return "The aircraft is accelerating down the runway and has not yet begun climbing."
        return "The aircraft appears to be in level flight, not actively climbing."

    if intent == "flight_phase":
        phase_descriptions = {
            "engine_off": "parked with the engine off",
            "taxi": "taxiing on the ground",
            "takeoff_roll": "on the takeoff roll, accelerating down the runway",
            "climb": "climbing",
            "level_flight": "in level cruise flight",
        }
        return f"Based on current readings, the aircraft appears to be {phase_descriptions.get(phase, 'in an unclear phase')}."

    return "I recognized the topic but don't have a specific answer template for it yet."


# =========================================================================
# MAIN
# =========================================================================
print(f"Running perception on: {args.image}")
params = run_perception(args.image)
phase = infer_phase(params)

print(f"\nPredicted parameters:")
print(f"  altitude: {params['altitude']:.0f} ft")
print(f"  airspeed: {params['airspeed']:.0f} kt")
print(f"  heading: {params['heading']:.0f} deg")
print(f"  pitch_bin: {params['pitch_bin']} (~{params['pitch_value']:.0f} deg)")
print(f"  roll_bin: {params['roll_bin']} (~{params['roll_value']:.0f} deg)")
print(f"  inferred phase (heuristic, not a trained classifier): {phase}")

if args.questions_file:
    with open(args.questions_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    print(f"\nRunning {len(questions)} questions from {args.questions_file}:\n")
    for q in questions:
        intent, score = route_question(q)
        answer = generate_answer(intent, params, phase, score)
        print(f"Q: {q}")
        print(f"A: {answer}")
        print(f"  (matched intent: {intent}, confidence: {score:.2f})\n")

elif args.question:
    intent, score = route_question(args.question)
    answer = generate_answer(intent, params, phase, score)
    print(f"\nQ: {args.question}")
    print(f"A: {answer}")
    print(f"  (matched intent: {intent}, confidence: {score:.2f})")
else:
    print("\nEnter questions (blank line to quit):")
    while True:
        q = input("Q: ").strip()
        if not q:
            break
        intent, score = route_question(q)
        answer = generate_answer(intent, params, phase, score)
        print(f"A: {answer}")
        print(f"  (matched intent: {intent}, confidence: {score:.2f})\n")
