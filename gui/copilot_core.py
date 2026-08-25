r"""
copilot_core.py
Shared core logic for the Visual Co-Pilot VQA system. Used by both
vqa_system.py (CLI) and app.py (Gradio UI) so the two never drift out
of sync — this was split out after a real bug where the CLI wrapper's
preprocessing didn't match the trained model (letterbox vs plain resize),
which silently degraded predictions until manually diagnosed.

This module has NO argparse / top-level side effects — it's a plain
importable library. All configuration is passed as function arguments.
"""

import os
import json
import time
import math

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image


# =========================================================================
# MODEL DEFINITION
# =========================================================================
class LetterboxResize:
    """Scale uniformly (no distortion) then pad to target size."""
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


class CopilotSystem:
    """Loads the trained model once, exposes perception + phase inference +
    question routing + answer generation. Instantiate once, reuse across
    many requests (important for the Gradio UI, which stays running)."""

    def __init__(self, data_dir, img_width=224, img_height=224, use_letterbox=False,
                 confidence_threshold=0.35):
        self.data_dir = data_dir
        self.model_dir = os.path.join(data_dir, "model_output")
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(self.model_dir, "norm_stats.json")) as f:
            self.norm_stats = json.load(f)
        with open(os.path.join(self.model_dir, "label_maps.json")) as f:
            self.label_maps = json.load(f)

        self.pitch_labels = self.label_maps["pitch_labels"]
        self.roll_labels = self.label_maps["roll_labels"]
        self.pitch_edges = self.label_maps["pitch_bin_edges"]
        self.roll_edges = self.label_maps["roll_bin_edges"]

        self.model = CockpitNet(len(self.pitch_labels), len(self.roll_labels)).to(self.device)
        checkpoint = torch.load(os.path.join(self.model_dir, "best_model.pt"), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.checkpoint_epoch = checkpoint.get("epoch")

        if use_letterbox:
            resize_step = LetterboxResize(img_width, img_height)
        else:
            resize_step = transforms.Resize((img_height, img_width))

        self.transform = transforms.Compose([
            resize_step,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Intent router — lazy-loaded so this module can be imported even
        # without sentence-transformers installed (falls back to keyword match)
        self._embedder = None
        self._intent_embeddings = None
        self._intent_labels = None
        self._init_router()

    # ---------------------------------------------------------------
    def _init_router(self):
        self.intent_examples = {
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
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            texts, labels = [], []
            for intent, examples in self.intent_examples.items():
                for ex in examples:
                    texts.append(ex)
                    labels.append(intent)
            self._intent_embeddings = self._embedder.encode(texts, convert_to_tensor=True)
            self._intent_labels = labels
            self.have_embedder = True
        except ImportError:
            self.have_embedder = False

    # ---------------------------------------------------------------
    def run_perception(self, image_path_or_pil):
        """Image -> dict of predicted flight parameters. Also returns
        per-stage timing in milliseconds for latency reporting."""
        timings = {}

        t0 = time.time()
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil).convert("RGB")
        else:
            img = image_path_or_pil.convert("RGB")
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        timings["preprocess_ms"] = (time.time() - t0) * 1000

        t0 = time.time()
        with torch.no_grad():
            pred = self.model(img_t)
        timings["inference_ms"] = (time.time() - t0) * 1000

        alt_stats, spd_stats = self.norm_stats["altitude"], self.norm_stats["airspeed"]
        altitude = pred["altitude"].item() * alt_stats["std"] + alt_stats["mean"]
        airspeed = pred["airspeed"].item() * spd_stats["std"] + spd_stats["mean"]

        sin_h, cos_h = pred["heading_sincos"][0].tolist()
        heading = math.degrees(math.atan2(sin_h, cos_h)) % 360

        pitch_idx = pred["pitch_logits"].argmax(1).item()
        roll_idx = pred["roll_logits"].argmax(1).item()
        pitch_bin = self.pitch_labels[pitch_idx]
        roll_bin = self.roll_labels[roll_idx]
        pitch_mid = (self.pitch_edges[pitch_idx] + self.pitch_edges[pitch_idx + 1]) / 2
        roll_mid = (self.roll_edges[roll_idx] + self.roll_edges[roll_idx + 1]) / 2

        params = {
            "altitude": altitude,
            "airspeed": airspeed,
            "heading": heading,
            "pitch_bin": pitch_bin,
            "roll_bin": roll_bin,
            "pitch_value": pitch_mid,
            "roll_value": roll_mid,
        }
        return params, timings

    # ---------------------------------------------------------------
    def infer_phase(self, params):
        """Heuristic phase context — NOT a trained classifier. Kept in
        sync with full_test_evaluation.py; if you change one, change both."""
        alt = params["altitude"]
        spd = params["airspeed"]
        pitch = params["pitch_value"]

        if alt < 150:
            if spd < 10:
                return "engine_off"
            elif spd < 40:
                return "taxi"
            else:
                return "takeoff_roll"
        elif alt < 500:
            if pitch >= 3:
                return "climb"
            else:
                return "takeoff_roll"
        elif pitch >= 3 and alt < 4500:
            return "climb"
        else:
            return "level_flight"

    # ---------------------------------------------------------------
    def route_question(self, question):
        t0 = time.time()
        if not self.have_embedder:
            q = question.lower()
            for intent, examples in self.intent_examples.items():
                for ex in examples:
                    if any(word in q for word in ex.split()):
                        return intent, 1.0, (time.time() - t0) * 1000
            return None, 0.0, (time.time() - t0) * 1000

        from sentence_transformers import util
        q_embedding = self._embedder.encode(question, convert_to_tensor=True)
        similarities = util.cos_sim(q_embedding, self._intent_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        best_intent = self._intent_labels[best_idx]
        routing_ms = (time.time() - t0) * 1000
        return best_intent, best_score, routing_ms

    # ---------------------------------------------------------------
    def generate_answer(self, intent, params, phase, score):
        if intent is None or score < self.confidence_threshold:
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

    # ---------------------------------------------------------------
    def answer(self, image_path_or_pil, question):
        """One-shot convenience: image + question -> full result dict,
        including timing breakdown for latency reporting."""
        params, perception_timings = self.run_perception(image_path_or_pil)
        phase = self.infer_phase(params)
        intent, score, routing_ms = self.route_question(question)
        answer_text = self.generate_answer(intent, params, phase, score)

        total_ms = perception_timings["preprocess_ms"] + perception_timings["inference_ms"] + routing_ms

        return {
            "params": params,
            "phase": phase,
            "intent": intent,
            "confidence": score,
            "answer": answer_text,
            "timings": {
                "preprocess_ms": perception_timings["preprocess_ms"],
                "inference_ms": perception_timings["inference_ms"],
                "routing_ms": routing_ms,
                "total_ms": total_ms,
            },
        }
