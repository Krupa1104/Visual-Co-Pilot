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
        # without the LLM gate's dependencies installed (falls back to
        # TF-IDF, then a trivial substring matcher).
        self._init_router()

    # ---------------------------------------------------------------
    # Short, plain-language descriptions of each real intent, used only to
    # build the LLM gate's system prompt below. Unlike the old approach
    # (embedding/TF-IDF similarity against dozens of hand-written example
    # phrasings, hand-tuned stopwords, etc.), the LLM gate needs no example
    # bank at all — it reasons directly from these one-line descriptions,
    # which is what actually eliminates the whack-a-mole tuning loop.
    INTENT_DESCRIPTIONS = {
        "get_altitude": "the question asks about altitude, height, or elevation",
        "get_airspeed": "the question asks about airspeed, speed, or velocity",
        "get_heading": "the question asks about heading, compass direction, or bearing",
        "is_banking": "the question asks about banking, rolling, turning, or wing tilt",
        "is_climbing": "the question asks about climbing, ascending, or gaining altitude",
        "flight_phase": "the question asks about the overall flight phase, status, "
                         "or what the aircraft is currently doing",
    }

    def _build_llm_system_prompt(self):
        lines = [
            "You are a strict classifier for a cockpit flight-instrument "
            "question-answering assistant.",
            "Given a user's question, decide which ONE of the following "
            "categories it belongs to, and respond with EXACTLY ONE WORD: "
            "the category name, and nothing else — no punctuation, no "
            "explanation, no extra words.",
            "",
        ]
        for label, desc in self.INTENT_DESCRIPTIONS.items():
            lines.append(f"{label}: {desc}")
        lines.append(
            "IRRELEVANT: the question is not about any of the above — e.g. "
            "weather, small talk, general knowledge, or anything unrelated "
            "to this aircraft's instruments"
        )
        lines.append("")
        lines.append("Respond with only the single category word.")
        return "\n".join(lines)

    def _init_router(self):
        # Kept ONLY as material for the TF-IDF / trivial-substring fallback
        # tiers below — used if the LLM gate can't load (e.g. no network on
        # first run to download the model). The LLM gate itself doesn't use
        # this at all.
        self.intent_examples = {
            "get_altitude": [
                "what is the altitude", "how high are we", "how high are we flying",
                "current elevation", "altimeter reading", "what altitude are we at",
                "what's the plane's altitude right now",
                "how many feet are we at", "what does the altimeter say",
                "how high up are we", "what is our current height above ground",
            ],
            "get_airspeed": [
                "what is the airspeed", "how fast are we going", "current speed",
                "what speed are we flying at", "airspeed indicator reading",
                "what's the current velocity",
                "how fast is the aircraft traveling", "what does the airspeed indicator show",
                "how many knots are we flying at",
            ],
            "get_heading": [
                "what heading are we on", "which direction are we flying",
                "current compass heading", "what direction is the plane pointed",
                "what's our heading", "which way are we headed",
                "what bearing are we flying",
                "what direction is the nose pointed",
                "what degrees on the compass are we at",
            ],
            "is_banking": [
                "are we banking", "is the plane turning", "are we tilted left or right",
                "is the aircraft rolling", "how much are we banked",
                "are the wings level", "is the plane rolled to one side",
                "are we in a turn right now", "how steep is our bank angle",
                "is the aircraft leaning left or right", "are we rolling left or right",
                "what's our current bank angle",
            ],
            "is_climbing": [
                "are we climbing", "are we gaining altitude", "is the nose up",
                "are we ascending", "is the aircraft climbing",
                "are we going up", "is the plane gaining height",
                "are we in a climb right now", "is our altitude increasing",
                "is the nose pitched up", "are we still ascending",
            ],
            "flight_phase": [
                "what phase of flight are we in",
                "are we taking off or landing", "what's our current status",
                "what stage of the flight is this",
                "are we on the ground or in the air",
                "describe our current flight phase", "is this takeoff, climb, or cruise",
                "what mode of flight is the aircraft in",
            ],
        }

        # --- Tier 1 (best): local LLM relevance + intent gate -----------
        # A small instruct model classifies the question directly against
        # plain-language category descriptions — real language
        # understanding, not pattern matching against canned phrasings.
        # This is the tier the router redesign is centered on.
        self.have_llm_gate = False
        self.have_embedder = False  # retained only so existing callers
                                     # (app.py/vqa_system.py/evaluate_router.py)
                                     # that check this flag don't break; the
                                     # embedding-based router has been
                                     # removed in favor of the LLM gate.
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            self._llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            self._llm_model.to(self.device)
            self._llm_model.eval()
            self._llm_system_prompt = self._build_llm_system_prompt()
            self.have_llm_gate = True
        except ImportError:
            print("WARNING: transformers not installed — falling back to TF-IDF matching.")
            self.have_llm_gate = False
        except Exception as e:
            # Most commonly: no network access on first run, since this
            # downloads ~3GB from Hugging Face and caches it locally
            # afterward (same failure mode the old sentence-transformers
            # embedder hit). Logged, not silent, and doesn't crash startup.
            print(f"WARNING: local LLM gate failed to load "
                  f"({type(e).__name__}: {e}) — falling back to TF-IDF matching.")
            self.have_llm_gate = False

        # --- Tier 2: TF-IDF + cosine similarity (no network needed) ------
        # Used only if the LLM gate couldn't load. Uses sklearn's built-in
        # `stop_words="english"` list (a standard, pre-vetted list — not
        # something invented here) plus IDF weighting, which automatically
        # down-weights any word that's common across many example phrases.
        self.have_tfidf = False
        if not self.have_llm_gate:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                texts, labels = [], []
                for intent, examples in self.intent_examples.items():
                    for ex in examples:
                        texts.append(ex)
                        labels.append(intent)
                self._tfidf_vectorizer = TfidfVectorizer(
                    stop_words="english", ngram_range=(1, 2), min_df=1)
                self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
                self._tfidf_labels = labels
                self.have_tfidf = True
            except ImportError:
                print("WARNING: scikit-learn not installed either — falling back "
                      "to a trivial substring matcher. Question routing quality "
                      "will be poor; get the LLM gate or scikit-learn working "
                      "for real matching.")
                self.have_tfidf = False

    # ---------------------------------------------------------------
    def _llm_classify(self, question):
        """Returns (intent_or_None, raw_model_output_text)."""
        messages = [
            {"role": "system", "content": self._llm_system_prompt},
            {"role": "user", "content": question},
        ]
        prompt = self._llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._llm_tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self._llm_model.generate(
                **inputs, max_new_tokens=6, do_sample=False,
                pad_token_id=self._llm_tokenizer.eos_token_id,
            )
        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        gen_text = self._llm_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        gen_lower = gen_text.lower()
        for label in list(self.INTENT_DESCRIPTIONS.keys()) + ["irrelevant"]:
            if label in gen_lower:
                return (None if label == "irrelevant" else label), gen_text
        return "PARSE_FAILED", gen_text

    def _keyword_edge_case_fallback(self, question):
        """Only invoked when the LLM gate is active but returned an
        unparseable response — rare, since generation is constrained to 6
        tokens with greedy decoding. This is a minimal safety net so one bad
        generation doesn't crash the pipeline; it is deliberately NOT the
        primary routing logic and isn't meant to be expanded/tuned over
        time the way the old keyword approach was."""
        q = question.lower()
        checks = {
            "get_altitude": ["altitude", "height", "elevation", "altimeter"],
            "get_airspeed": ["airspeed", "speed", "knots", "velocity"],
            "get_heading": ["heading", "direction", "compass", "bearing"],
            "is_banking": ["bank", "roll", "tilt", "turn"],
            "is_climbing": ["climb", "ascend", "ascending", "gaining"],
            "flight_phase": ["phase", "status", "takeoff", "landing", "cruise"],
        }
        for intent, words in checks.items():
            if any(w in q for w in words):
                return intent
        return None

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
    @staticmethod
    def _trivial_tokens(text):
        import re
        return set(re.findall(r"[a-z]+", text.lower()))

    def route_question(self, question):
        """Returns (intent, score, routing_ms). `intent` is one of the real
        intent names, or None if nothing matched (irrelevant, or nothing
        matched at all). Three router qualities, in preference order: local
        LLM relevance/intent gate (best — real language understanding,
        needs the model downloaded/cached once) > TF-IDF cosine similarity
        (good, no network needed) > trivial substring overlap (last resort,
        only if neither of the above is available — logged as a real
        quality warning, not silently used)."""
        t0 = time.time()

        if self.have_llm_gate:
            label, raw_output = self._llm_classify(question)
            routing_ms = (time.time() - t0) * 1000
            if label == "PARSE_FAILED":
                label = self._keyword_edge_case_fallback(question)
                return label, (0.5 if label else 0.0), routing_ms
            return label, (1.0 if label else 0.0), routing_ms

        if self.have_tfidf:
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = self._tfidf_vectorizer.transform([question])
            sims = cosine_similarity(q_vec, self._tfidf_matrix)[0]
            best_idx = sims.argmax()
            best_score = float(sims[best_idx])
            best_intent = self._tfidf_labels[best_idx]
            if best_score <= 0:
                return None, 0.0, (time.time() - t0) * 1000
            return best_intent, best_score, (time.time() - t0) * 1000

        # Last resort only: neither the LLM gate nor scikit-learn available.
        # No stopword tuning here on purpose — this path is explicitly
        # low-quality and flagged as such at construction time, rather than
        # incrementally patched to paper over its weaknesses.
        q_words = self._trivial_tokens(question)
        best_intent, best_score = None, 0.0
        for intent, examples in self.intent_examples.items():
            for ex in examples:
                ex_words = self._trivial_tokens(ex)
                if not ex_words:
                    continue
                score = len(q_words & ex_words) / len(ex_words)
                if score > best_score:
                    best_score = score
                    best_intent = intent
        if best_intent is None:
            return None, 0.0, (time.time() - t0) * 1000
        return best_intent, best_score, (time.time() - t0) * 1000

    # ---------------------------------------------------------------
    NOT_APPLICABLE = (
        "Not applicable — I couldn't confidently match that question to something "
        "I can answer. I can currently answer questions about altitude, airspeed, "
        "heading, banking/roll, climbing, and general flight phase."
    )

    def generate_answer(self, intent, params, phase, score):
        # Simple error/fallback case: no intent matched at all, or the best
        # match scored below confidence_threshold (this is also how
        # off-topic questions get caught — they're just real intents that
        # happen to score low, not a separately detected category).
        if intent is None or score < self.confidence_threshold:
            return self.NOT_APPLICABLE

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
        including timing breakdown for latency reporting.

        NOTE: compound/multi-part questions (e.g. "what's our altitude and
        heading?") are NOT handled — this is the Section 7 Task 1 stretch
        goal and remains unimplemented. Such a question is routed to
        whichever single intent scores highest, and only that one part is
        answered. Flagging this explicitly rather than pretending it's
        handled."""
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
