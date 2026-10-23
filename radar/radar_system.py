r"""
radar_system.py
Orchestration layer for the Radar VQA system. Sits ON TOP of the real,
already-committed Radar_Phase1 repo (radar_generation/, etc.) for the
callsign locator, and reimplements the CNN VQA model's LEGACY architecture
directly in this file -- see the note below on why.

Adds the layer that exists only as notebook cells today (Radar_Cnn_Review1_
Phase3_2.ipynb, Steps 16-20): YOLOv5 aircraft-blip detection used to cross-
check the CNN's counting answers, a callsign flight-locator, an LLM
relevance gate, and confidence-based hedge/decline logic -- wrapped as a
plain class with no Colab-only APIs, the same role copilot_core.py plays
for the cockpit module.

WHY THE CNN MODEL IS REIMPLEMENTED HERE INSTEAD OF IMPORTED FROM models/:
The repo's models/cnn_model.py was edited to a newer architecture (bigger
embed/hidden dims, BatchNorm1d, added self-attention) after cnn_best_model.pth
was trained, and no checkpoint matching that newer architecture exists in
Drive. Loading cnn_best_model.pth against the CURRENT models/cnn_model.py
throws a state_dict mismatch. The classes below match what cnn_best_model.pth
was ACTUALLY trained with (confirmed by two separate mismatch tracebacks),
so this loads correctly today. If models/cnn_model.py and its checkpoint
are ever brought back in sync (retrain + re-save, or revert the class),
swap this back to `from models.cnn_inference import CNNInferenceEngine`.

REQUIRED: this file must live at the ROOT of your Radar_Phase1 checkout
(next to the radar_generation/ etc. folders), or you must pass repo_root
pointing at that checkout, so `from radar_generation.radar_renderer import
RadarRenderer` resolves to YOUR real code.

RUNTIME ARTIFACTS THIS MODULE NEEDS AND DOES NOT PROVIDE (from Drive):
  <base_dir>/models/cnn_best_model.pth
  <base_dir>/data/processed/answer_index.json
  <base_dir>/models/yolov5_aircraft/weights/best.pt
  <base_dir>/data/raw/rendered_frames.json
  <base_dir>/data/raw/images/
"""

import os
import re
import sys
import json
import difflib
import pathlib
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image, ImageDraw
from ultralytics import YOLO

# =========================================================================
# LEGACY CNN VQA architecture -- matches cnn_best_model.pth's ACTUAL saved
# weights (confirmed by two separate state_dict mismatches against the repo's
# current models/cnn_model.py, which was edited after this checkpoint was
# trained: embed 128 vs 64, hidden 512 vs 256, 3 GRU layers vs 2, BatchNorm1d
# vs LayerNorm, plus an added self-attention block). Defined here rather than
# imported from models/cnn_model.py so loading actually succeeds against the
# checkpoint you have, instead of the newer architecture in git that has no
# matching trained weights in Drive.
# =========================================================================
class _LegacyCNNVisualEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()

        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1
        )

        self.features = nn.Sequential(*list(backbone.children())[:-1])

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, out_dim),
            nn.BatchNorm1d(out_dim),   # <-- CHANGED from LayerNorm
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x):
        x = self.features(x)
        return self.proj(x)


class _LegacyQuestionEncoder(nn.Module):
    def __init__(self, vocab=40, embed=128, hidden=512, out_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab + 1, embed, padding_idx=0)
        self.gru = nn.GRU(
            embed,
            hidden,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x):
        emb = self.embed(x)
        out, h = self.gru(emb)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(h)


class _LegacyCNNVQAModel(nn.Module):
    def __init__(self, num_classes, device="cpu", feat_dim=512):
        super().__init__()
        self.device = device
        self.visual_enc = _LegacyCNNVisualEncoder(out_dim=feat_dim)
        self.question_enc = _LegacyQuestionEncoder(out_dim=feat_dim)
        self.self_attn = nn.MultiheadAttention(feat_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(feat_dim, num_heads=8,
                                                 dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        self.to(device)

    def forward(self, images, questions):
        images = images.to(self.device)
        questions = questions.to(self.device)
        v = self.visual_enc(images)
        q = self.question_enc(questions)
        v_s = v.unsqueeze(1)
        q_s = q.unsqueeze(1)
        att, _ = self.cross_attn(v_s, q_s, q_s)
        fused = self.norm1(v_s + att)
        self_att, _ = self.self_attn(fused, fused, fused)
        fused = self.norm2(fused + self_att).squeeze(1)
        return self.classifier(torch.cat([fused, q], dim=-1))

    def predict(self, images, questions, answer_index):
        self.eval()
        with torch.no_grad():
            logits = self.forward(images, questions)
            probs = F.softmax(logits, dim=-1)
            ids = logits.argmax(-1).cpu().tolist()
            confs = probs.max(-1).values.cpu().tolist()
        idx2ans = {v: k for k, v in answer_index.items()}
        return [idx2ans.get(i, "unknown") for i in ids], confs

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return self


class _LegacyCNNInferenceEngine:
    CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789 ?,'.")
    CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
    MAX_Q = 60

    def __init__(self, model_path, answer_index_path, device="cpu"):
        self.device = device
        with open(answer_index_path) as f:
            self.answer_index = json.load(f)
        self.model = _LegacyCNNVQAModel(len(self.answer_index), device)
        self.model.load(model_path)
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def answer(self, image, question):
        img_t = self.tf(image.convert("RGB")).unsqueeze(0)
        q = question.lower()[: self.MAX_Q]
        q_tok = [self.CHAR2IDX.get(c, 0) for c in q]
        q_tok += [0] * (self.MAX_Q - len(q_tok))
        q_t = torch.tensor([q_tok], dtype=torch.long)
        answers, confs = self.model.predict(img_t, q_t, self.answer_index)
        return {
            "question": question,
            "answer": answers[0],
            "confidence": round(confs[0], 4),
            "route": "cnn_pixel_reading",
            "source": "resnet50_cnn",
        }


class RadarSystem:
    """Loads the real CNN VQA engine + fine-tuned YOLOv5 detector once,
    builds the callsign index, and exposes classify_relevance /
    locate_and_highlight / evaluate_and_answer / handle_question.

    Pass an already-loaded (llm_tokenizer, llm_model) pair to share ONE
    Qwen2.5-1.5B-Instruct instance with the cockpit module's CopilotSystem
    instead of loading a second ~3GB copy. Leaving both None loads its own
    copy -- fine for running/testing this module standalone.
    """

    CONF_HEDGE_THRESHOLD = 0.55
    CONF_DECLINE_THRESHOLD = 0.30
    YOLO_COUNT_TOLERANCE = 1

    RELEVANCE_SYSTEM_PROMPT = (
        "You are a strict relevance filter in front of an aviation radar "
        "Visual Question Answering system. The system only sees a single "
        "224x224 synthetic PPI radar image containing a handful of aircraft "
        '"blips" -- it can answer questions about: how many aircraft are '
        "visible, an aircraft's position (left/right/top/bottom/quadrant/"
        "center), altitude, heading, callsign, comparisons between aircraft "
        "(highest/lowest altitude etc.), and locating/highlighting a "
        "specific aircraft by callsign.\n\n"
        "It CANNOT answer anything requiring outside knowledge, real-world "
        "facts, opinions, math unrelated to the image, or general "
        "conversation.\n\n"
        "Reply with exactly one word: RELEVANT or IRRELEVANT. No "
        "punctuation, no explanation."
    )

    _RELEVANCE_KEYWORDS = [
        "aircraft", "flight", "blip", "radar", "altitude", "heading", "callsign",
        "left", "right", "top", "bottom", "center", "quadrant", "how many",
        "locate", "find", "highlight", "position", "coordinate",
    ]

    LOCATE_TRIGGER = re.compile(r"\b(locate|find|where\s+is|highlight|show\s+me)\b", re.IGNORECASE)
    CALLSIGN_PATTERN = re.compile(r"\b[A-Za-z]{2,5}\d{1,5}[A-Za-z]?\b")
    COUNT_QUESTION = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b", re.IGNORECASE)

    def __init__(self, base_dir, repo_root=None, yolo_root=None, yolo_conf=0.25,
                 device=None, llm_tokenizer=None, llm_model=None,
                 hedge_threshold=None, decline_threshold=None, count_tolerance=None,
                 cnn_checkpoint="cnn_best_model.pth"):
        self.cnn_checkpoint = cnn_checkpoint
        self.base_dir = base_dir
        self.proc_dir = os.path.join(base_dir, "data", "processed")
        self.raw_dir = os.path.join(base_dir, "data", "raw")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Make sure the real repo (models/, radar_generation/) is importable.
        self.repo_root = repo_root or os.getcwd()
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)

        self.hedge_threshold = hedge_threshold if hedge_threshold is not None else self.CONF_HEDGE_THRESHOLD
        self.decline_threshold = decline_threshold if decline_threshold is not None else self.CONF_DECLINE_THRESHOLD
        self.count_tolerance = count_tolerance if count_tolerance is not None else self.YOLO_COUNT_TOLERANCE

        self.log_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.retrain_log = os.path.join(self.log_dir, "low_confidence_for_retraining.jsonl")
        self.escalation_log = os.path.join(self.log_dir, "escalations.jsonl")

        self._load_cnn_engine()
        self.yolo_root = yolo_root or os.path.join(os.path.dirname(base_dir), "yolov5")
        self._load_yolo(yolo_conf=yolo_conf)
        self._load_renderer()
        self._build_callsign_index()
        self._init_relevance_llm(llm_tokenizer, llm_model)

    # ---------------------------------------------------------------
    def _load_cnn_engine(self):
        cnn_best = os.path.join(self.base_dir, "models", self.cnn_checkpoint)
        answer_index_path = os.path.join(self.proc_dir, "answer_index.json")
        if not os.path.exists(cnn_best):
            raise FileNotFoundError(
                f"Radar CNN checkpoint not found at {cnn_best}. Copy "
                "cnn_best_model.pth from the aviation_vqa_output Drive "
                "folder's models/ into <base_dir>/models/.")
        if not os.path.exists(answer_index_path):
            raise FileNotFoundError(
                f"answer_index.json not found at {answer_index_path}. Copy "
                "it from <drive_base_dir>/data/processed/answer_index.json.")

        # Uses the LEGACY architecture defined at the top of this file
        # (matches cnn_best_model.pth's actual saved weights), NOT
        # models/cnn_inference.py -- that imports the repo's current
        # models/cnn_model.py, which has since been edited to a different
        # architecture with no matching trained checkpoint in Drive.
        self.cnn_engine = _LegacyCNNInferenceEngine(cnn_best, answer_index_path, device=self.device)

    # ---------------------------------------------------------------
    def _load_yolo(self, yolo_conf=0.25):
        """Load YOLOv5 from a local repository using torch.hub."""

        weights = os.path.join(
            self.base_dir, "models", "yolov5_aircraft", "weights", "best.pt"
        )
        if not os.path.exists(weights):
            raise FileNotFoundError(f"YOLO weights not found at {weights}")

        if not self.yolo_root or not os.path.exists(self.yolo_root):
            raise FileNotFoundError(
                f"YOLOv5 repo not found at {self.yolo_root}. Clone https://github.com/ultralytics/yolov5"
            )

        # Windows fix for YOLOv5 checkpoints trained on Linux/Colab
        _original_posix = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

        try:
            self.yolo_model = torch.hub.load(
                self.yolo_root,
                "custom",
                path=weights,
                source="local",
                force_reload=True,
            )
        finally:
            pathlib.PosixPath = _original_posix

        self.yolo_model.conf = yolo_conf
        

    def _load_renderer(self):
        # Your real RadarRenderer, imported directly -- used only for its
        # get_norm_coords() math here, no rendering happens at inference time.
        from radar_generation.radar_renderer import RadarRenderer
        self._renderer_for_coords = RadarRenderer(output_dir=self.raw_dir)

    # ---------------------------------------------------------------
    def _build_callsign_index(self):
        rendered_path = os.path.join(self.raw_dir, "rendered_frames.json")
        if not os.path.exists(rendered_path):
            raise FileNotFoundError(
                f"rendered_frames.json not found at {rendered_path}. Copy it "
                "from <drive_base_dir>/data/raw/rendered_frames.json, along "
                "with the matching images/ folder.")
        with open(rendered_path) as f:
            raw_rendered = json.load(f)

        # The JSON's stored "image_path" fields were baked in wherever this
        # was originally rendered (e.g. a Colab /content/drive/... path) --
        # they won't exist on this machine. attach_paths() ignores the
        # stored path and rebuilds it from frame_id + the LOCAL images/
        # folder instead, so this works regardless of where the manifest
        # was first generated.
        image_dir = os.path.join(self.raw_dir, "images")
        self.rendered = self._renderer_for_coords.attach_paths(raw_rendered, image_dir)

        self.callsign_index = {}
        for frame in self.rendered:
            if not os.path.exists(frame["image_path"]):
                continue
            enriched = self._renderer_for_coords.get_norm_coords(frame["aircraft"])
            for ac in enriched:
                cs = ac["callsign"].strip().upper()
                self.callsign_index.setdefault(cs, []).append({
                    "frame_id": frame["frame_id"],
                    "image_path": frame["image_path"],
                    "callsign": ac["callsign"],
                    "icao24": ac.get("icao24"),
                    "latitude": ac["latitude"],
                    "longitude": ac["longitude"],
                    "altitude": ac.get("altitude"),
                    "heading": ac.get("heading"),
                    "nx": ac["nx"], "ny": ac["ny"],
                })

    # ---------------------------------------------------------------
    def _init_relevance_llm(self, llm_tokenizer, llm_model):
        if llm_tokenizer is not None and llm_model is not None:
            self._llm_tokenizer = llm_tokenizer
            self._llm_model = llm_model
            self.have_llm_gate = True
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            self._llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            self._llm_model.to(self.device)
            self._llm_model.eval()
            self.have_llm_gate = True
        except ImportError:
            print("WARNING: transformers not installed -- relevance filtering "
                  "will use a keyword fallback instead of the LLM gate.")
            self.have_llm_gate = False
        except Exception as e:
            print(f"WARNING: relevance LLM failed to load ({type(e).__name__}: {e}) "
                  "-- falling back to keyword matching.")
            self.have_llm_gate = False

    # ---------------------------------------------------------------
    def _keyword_relevance_fallback(self, question):
        q = question.lower()
        return any(k in q for k in self._RELEVANCE_KEYWORDS)

    def classify_relevance(self, question):
        if not self.have_llm_gate:
            return self._keyword_relevance_fallback(question)
        messages = [
            {"role": "system", "content": self.RELEVANCE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = self._llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._llm_tokenizer(prompt, return_tensors="pt").to(self.device)
        try:
            with torch.no_grad():
                out = self._llm_model.generate(
                    **inputs, max_new_tokens=4, do_sample=False,
                    pad_token_id=self._llm_tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            verdict = self._llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()
            if "IRRELEVANT" in verdict:
                return False
            if "RELEVANT" in verdict:
                return True
            return self._keyword_relevance_fallback(question)
        except Exception as e:
            print(f"[relevance] LLM inference failed ({e}); using keyword fallback.")
            return self._keyword_relevance_fallback(question)

    # ---------------------------------------------------------------
    def extract_callsign(self, question):
        candidates = self.CALLSIGN_PATTERN.findall(question)
        if not candidates:
            return None
        for c in candidates:
            if c.upper() in self.callsign_index:
                return c.upper()
        for c in candidates:
            match = difflib.get_close_matches(c.upper(), self.callsign_index.keys(), n=1, cutoff=0.6)
            if match:
                return match[0]
        return None

    def locate_flight(self, callsign):
        matches = self.callsign_index.get(callsign.upper())
        return matches[0] if matches else None

    def highlight_flight(self, record, upscale=2, img_size=224):
        img = Image.open(record["image_path"]).convert("RGB")
        disp = img.resize((img_size * upscale, img_size * upscale))
        draw = ImageDraw.Draw(disp)

        results = self.yolo_model(record["image_path"], size=img_size)
        df = results.pandas().xyxy[0]

        target_x = record["nx"] * img_size
        target_y = (1 - record["ny"]) * img_size

        matched, best_dist = None, float("inf")
        for _, row in df.iterrows():
            cx = (float(row.xmin) + float(row.xmax)) / 2.0
            cy = (float(row.ymin) + float(row.ymax)) / 2.0
            distance_squared = (cx - target_x) ** 2 + (cy - target_y) ** 2
            if distance_squared < best_dist:
                best_dist, matched = distance_squared, row

        # Avoid highlighting an unrelated detection when the nearest blip is
        # too far from the callsign's expected normalized position.
        max_distance_squared = 25 ** 2  # 25 px on the 224x224 radar image
        if best_dist > max_distance_squared:
            matched = None

        for _, row in df.iterrows():
            x1, y1, x2, y2 = [v * upscale for v in (row.xmin, row.ymin, row.xmax, row.ymax)]
            is_match = matched is not None and row.equals(matched)
            color = "#ffcc00" if is_match else "#888888"
            width = 4 if is_match else 1
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            if is_match:
                draw.text((x1, max(0, y1 - 14)), f"\U0001F3AF {record['callsign']}", fill="#ffcc00")

        return disp, matched is not None

    def locate_and_highlight(self, question):
        callsign = self.extract_callsign(question)
        if not callsign:
            return None
        record = self.locate_flight(callsign)
        if not record:
            return {"found": False,
                    "answer": f"I don't have a record of a flight called {callsign}.",
                    "image": None}
        disp, matched_in_image = self.highlight_flight(record)
        coord_text = (
            f"{record['callsign']} is at latitude {record['latitude']:.4f}, "
            f"longitude {record['longitude']:.4f}, altitude {record['altitude']} ft, "
            f"heading {record['heading']}\u00b0."
        )
        if not matched_in_image:
            coord_text += " (YOLO didn't detect a matching blip on this image, showing its known position only.)"
        return {"found": True, "answer": coord_text, "image": disp, "record": record}

    # ---------------------------------------------------------------
    @staticmethod
    def _extract_int(text):
        m = re.search(r"\d+", str(text))
        return int(m.group()) if m else None

    def _yolo_count(self, image_path, conf=0.25):
        prev_conf = self.yolo_model.conf
        self.yolo_model.conf = conf
        try:
            results = self.yolo_model(image_path, size=224)
            return len(results.pandas().xyxy[0])
        finally:
            self.yolo_model.conf = prev_conf

    def _log_jsonl(self, path, record):
        record = {**record, "timestamp": datetime.datetime.utcnow().isoformat()}
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _answer_quadrant(self, image_path):
        """Count YOLO detections in each quadrant and report the maximum."""
        results = self.yolo_model(image_path, size=224)
        detections = results.pandas().xyxy[0]

        counts = {
            "top-left": 0,
            "top-right": 0,
            "bottom-left": 0,
            "bottom-right": 0,
        }

        # The model input is 224x224, so the image midpoint is (112, 112).
        for _, row in detections.iterrows():
            cx = (float(row.xmin) + float(row.xmax)) / 2.0
            cy = (float(row.ymin) + float(row.ymax)) / 2.0
            horizontal = "left" if cx < 112 else "right"
            vertical = "top" if cy < 112 else "bottom"
            counts[f"{vertical}-{horizontal}"] += 1

        highest = max(counts.values())
        if highest == 0:
            return "I couldn't detect any aircraft confidently in this radar image."

        winners = [quadrant for quadrant, count in counts.items() if count == highest]
        if len(winners) > 1:
            details = ", ".join(f"{quadrant}: {counts[quadrant]}" for quadrant in winners)
            return (
                f"There is a tie between {details}. "
                "No single quadrant has the most detected aircraft."
            )

        winner = winners[0]
        return f"The {winner} quadrant has the most aircraft ({highest} detected)."

    def _answer_altitude_extreme_callsign(self, image_path, question):
        """Answer highest/lowest-altitude callsign questions from frame metadata."""
        q = question.lower()
        want_lowest = bool(re.search(r"\b(lowest|least|minimum|min)\b", q))
        want_highest = bool(re.search(r"\b(highest|highest-altitude|most)\b", q))
        if not (want_lowest or want_highest) or "altitude" not in q:
            return None

        # Only use metadata for the image currently being queried. Match full
        # paths first, then basename as Gradio may copy an image to a temp path.
        requested_path = os.path.normcase(os.path.abspath(str(image_path)))
        requested_name = os.path.normcase(os.path.basename(str(image_path)))
        matching_frames = []
        for frame in self.rendered:
            frame_path = str(frame.get("image_path", ""))
            if not frame_path:
                continue
            normalized = os.path.normcase(os.path.abspath(frame_path))
            if normalized == requested_path:
                matching_frames.append(frame)
                break
        if not matching_frames:
            matching_frames = [
                frame for frame in self.rendered
                if os.path.normcase(os.path.basename(str(frame.get("image_path", "")))) == requested_name
            ]
        if not matching_frames:
            return (
                "I can't reliably identify the highest/lowest-altitude callsign "
                "because this image doesn't match a frame in the radar metadata."
            )

        aircraft = matching_frames[0].get("aircraft", [])
        candidates = []
        for ac in aircraft:
            callsign = str(ac.get("callsign", "")).strip()
            altitude = ac.get("altitude")
            if not callsign or altitude is None:
                continue
            try:
                # Accept numeric values and numeric strings, but skip missing or invalid data.
                altitude_value = float(altitude)
            except (TypeError, ValueError):
                continue
            candidates.append((callsign, altitude_value))

        if not candidates:
            return "The radar metadata for this image has no usable callsign and altitude values."

        extreme_altitude = (min if want_lowest and not want_highest else max)(
            altitude for _, altitude in candidates
        )
        winners = [
            callsign for callsign, altitude in candidates
            if altitude == extreme_altitude
        ]
        direction = "lowest" if want_lowest and not want_highest else "highest"
        if len(winners) > 1:
            return (
                f"The {direction}-altitude aircraft are tied at {extreme_altitude:g} ft: "
                + ", ".join(winners) + "."
            )
        return f"{winners[0]} has the {direction} altitude at {extreme_altitude:g} ft."

    def _frame_records_for_image(self, image_path):
        """Return metadata records belonging to the exact selected radar frame."""
        requested = os.path.normcase(os.path.abspath(str(image_path)))
        requested_name = os.path.normcase(os.path.basename(str(image_path)))
        frame = None
        for item in self.rendered:
            item_path = str(item.get("image_path", ""))
            if item_path and os.path.normcase(os.path.abspath(item_path)) == requested:
                frame = item
                break
        if frame is None:
            matches = [item for item in self.rendered
                       if os.path.normcase(os.path.basename(str(item.get("image_path", "")))) == requested_name]
            if len(matches) == 1:
                frame = matches[0]
        if frame is None:
            return []
        records = []
        for record in self.callsign_index.values():
            for rec in record:
                if (str(rec.get("frame_id")) == str(frame.get("frame_id")) and
                    os.path.normcase(os.path.basename(str(rec.get("image_path", "")))) ==
                    os.path.normcase(os.path.basename(str(frame.get("image_path", ""))))):
                    records.append(rec)
        return records

    def _yolo_aircraft_records(self, image_path, max_match_px=28):
        """Detect boxes with YOLO and associate boxes with same-frame metadata by position."""
        results = self.yolo_model(image_path, size=224)
        detections = results.pandas().xyxy[0]
        records = self._frame_records_for_image(image_path)
        output = []
        unmatched = list(records)
        for _, box in detections.iterrows():
            cx = (float(box.xmin) + float(box.xmax)) / 2.0
            cy = (float(box.ymin) + float(box.ymax)) / 2.0
            best = None
            best_distance = float("inf")
            for rec in unmatched:
                try:
                    rx = float(rec["nx"]) * 224
                    ry = (1.0 - float(rec["ny"])) * 224
                except (KeyError, TypeError, ValueError):
                    continue
                distance = ((cx-rx)**2 + (cy-ry)**2) ** 0.5
                if distance < best_distance:
                    best, best_distance = rec, distance
            matched = best if best is not None and best_distance <= max_match_px else None
            if matched is not None:
                unmatched.remove(matched)
            output.append({
            "xmin": float(box.xmin),
            "ymin": float(box.ymin),
            "xmax": float(box.xmax),
            "ymax": float(box.ymax),
            "cx": cx,
            "cy": cy,
            "confidence": float(box.confidence),
            "record": matched,
            "match_distance_px": best_distance if matched is not None else None,
        })
        return output

    # Screen regions for YOLO-box questions, in normalised coords (nx: 0 left -> 1 right,
    # ny: 0 bottom -> 1 top). Same boundaries the dataset's question generator used.
    _SIDE_REGIONS = [
        (re.compile(r"\bleft\s+(side|half|part)\b|\bon the left\b", re.I), "on the left side",
         lambda x, y: x <= 0.40),
        (re.compile(r"\bright\s+(side|half|part)\b|\bon the right\b", re.I), "on the right side",
         lambda x, y: x >= 0.60),
        (re.compile(r"\b(upper|top)\s+(portion|half|part|side)\b|\bat the top\b", re.I),
         "in the upper portion", lambda x, y: y >= 0.60),
        (re.compile(r"\b(lower|bottom)\s+(portion|half|part|side)\b|\bat the bottom\b", re.I),
         "in the lower portion", lambda x, y: y <= 0.40),
        (re.compile(r"\b(near|at|in|around) the cent(er|re)\b", re.I), "near the center",
         lambda x, y: 0.30 <= x <= 0.70 and 0.30 <= y <= 0.70),
    ]
    # Counts that depend on altitude ("flying above 30,000 feet"): YOLO can't read altitudes,
    # so these go to the CNN instead of being answered with the total box count.
    _ALTITUDE_COUNT_RE = re.compile(
        r"\b(feet|ft|altitude|flying above|flying below|higher than|lower than)\b", re.I)

    def _from_yolo(self, answer):
        """Mark an answer as computed purely from YOLO boxes (so its detector confidence applies)."""
        self._last_answer_source = "yolo"
        return answer

    def _answer_yolo_question(self, image_path, question):
        """Answer common detection, callsign, position and metadata comparison questions."""
        q = question.lower()
        aircraft = self._yolo_aircraft_records(image_path)

        # Average confidence of YOLO detections in this image. Only reported for answers
        # computed from the boxes themselves (counts, quadrants, regions) -- not for answers
        # read from the frame's metadata (callsigns, altitudes), where it would mislead.
        self._last_yolo_confidence = (
            sum(ac["confidence"] for ac in aircraft) / len(aircraft)
            if aircraft else None
        )
        self._last_answer_source = "metadata"

        if not aircraft:
            return self._from_yolo("YOLOv5 did not detect any aircraft in this image.")

        # Altitude-based counts -> CNN (YOLO boxes carry no altitude).
        if self.COUNT_QUESTION.search(q) and self._ALTITUDE_COUNT_RE.search(q):
            return None

        # Count detections directly from YOLO boxes.
        
        # Handle quadrant questions BEFORE general counting questions.
        # Otherwise "How many aircraft are in the upper-left quadrant?"
        # incorrectly returns the total number of aircraft.

        if re.search(r"\b(quadrant|which part|which area)\b", q) or re.search(
            r"\b(upper|lower|top|bottom)\s*[- ]\s*(left|right)\b", q
        ):
            counts = {
                "top-left": 0,
                "top-right": 0,
                "bottom-left": 0,
                "bottom-right": 0,
            }

            for ac in aircraft:
                horizontal = "left" if ac["cx"] < 112 else "right"
                vertical = "top" if ac["cy"] < 112 else "bottom"
                counts[f"{vertical}-{horizontal}"] += 1

            # Questions asking for a particular quadrant's count.
            quadrant_match = re.search(
                r"\b(upper|lower|top|bottom)\s*[- ]\s*(left|right)\b",
                q,
            )

            if quadrant_match and self.COUNT_QUESTION.search(q):
                vertical_word, horizontal = quadrant_match.groups()
                vertical = (
                    "top" if vertical_word in ("upper", "top") else "bottom"
                )
                quadrant = f"{vertical}-{horizontal}"
                count = counts[quadrant]
                return self._from_yolo(
                    f"There {'is' if count == 1 else 'are'} {count} "
                    f"{'aircraft' if count != 1 else 'aircraft'} "
                    f"in the {quadrant} quadrant."
                )

            # Questions asking which quadrant has the most aircraft.
            if re.search(r"\b(most|more|highest|maximum)\b", q):
                highest = max(counts.values())
                winners = [
                    name for name, count in counts.items()
                    if count == highest
                ]

                if len(winners) > 1:
                    return self._from_yolo("Tie between " + ", ".join(
                        f"{name} ({counts[name]})" for name in winners
                    ) + ".")

                return self._from_yolo(
                    f"The {winners[0]} quadrant has the most detected "
                    f"aircraft ({highest})."
                )

            return self._from_yolo("Radar quadrant counts: " + ", ".join(
                f"{name}: {count}" for name, count in counts.items()
            ) + ".")

        # Left/right side, upper/lower portion, near the center: count or yes/no from YOLO boxes.
        for pattern, label, inside in self._SIDE_REGIONS:
            if pattern.search(q):
                hits = [ac for ac in aircraft if inside(ac["cx"] / 224, 1 - ac["cy"] / 224)]
                if self.COUNT_QUESTION.search(q):
                    return self._from_yolo(f"YOLOv5 detected {len(hits)} aircraft {label}.")
                if re.match(r"^\s*(is|are)\s+there\b", q):
                    return self._from_yolo(
                        f"{'Yes' if hits else 'No'} -- YOLOv5 detected {len(hits)} aircraft {label}.")
                break

        # "Are there more than N aircraft?" -> compare with the YOLO box count.
        more = re.search(r"\bmore than (\d+)\b", q)
        if more and re.search(r"\b(aircraft|planes?|flights?|blips?)\b", q):
            n, limit = len(aircraft), int(more.group(1))
            return self._from_yolo(f"{'Yes' if n > limit else 'No'} -- YOLOv5 detected {n} aircraft.")

        # General aircraft count, only after quadrant-specific questions.
        if self.COUNT_QUESTION.search(q):
            return self._from_yolo(
                f"YOLOv5 detected {len(aircraft)} aircraft "
                "in this radar image."
            )


        # Highest/lowest altitude or callsign.
        if "altitude" in q and re.search(r"\b(highest|lowest|highest-altitude|lowest-altitude|maximum|minimum|most|least)\b", q):
            candidates = []
            for ac in aircraft:
                rec = ac["record"]
                if not rec or rec.get("altitude") is None or not str(rec.get("callsign", "")).strip():
                    continue
                try:
                    candidates.append((float(rec["altitude"]), str(rec["callsign"]).strip(), ac))
                except (TypeError, ValueError):
                    continue
            if not candidates:
                return "I detected aircraft, but couldn't reliably match their boxes to callsign/altitude metadata in this frame."
            lowest = bool(re.search(r"\b(lowest|minimum|least)\b", q)) and not bool(re.search(r"\b(highest|maximum|most)\b", q))
            extreme = min(v[0] for v in candidates) if lowest else max(v[0] for v in candidates)
            winners = [(alt, cs, ac) for alt, cs, ac in candidates if alt == extreme]
            direction = "lowest" if lowest else "highest"
            if len(winners) > 1:
                return f"The {direction}-altitude aircraft are tied at {extreme:g} ft: " + ", ".join(x[1] for x in winners) + "."
            return f"{winners[0][1]} has the {direction} altitude ({extreme:g} ft)."

        # A specific callsign: return attributes or location, based on same-frame match.
        requested_callsign = self.extract_callsign(question)
        if requested_callsign:
            match = next((ac for ac in aircraft if ac["record"] and
                          str(ac["record"].get("callsign", "")).upper() == requested_callsign.upper()), None)
            if match is None:
                return f"I couldn't match {requested_callsign} to a YOLOv5 detection in this image."
            rec = match["record"]
            if re.search(r"\b(highlight|show|locate|find|where|position|location)\b", q):
                disp, found = self.highlight_flight(rec)
                text = f"{rec['callsign']} is detected in the {('top' if match['cy'] < 112 else 'bottom')}-{'left' if match['cx'] < 112 else 'right'} area."
                if rec.get("altitude") is not None:
                    text += f" Altitude: {rec['altitude']} ft."
                if rec.get("heading") is not None:
                    text += f" Heading: {rec['heading']} degrees."
                if not found:
                    text += " The metadata position did not match a YOLO box closely enough to highlight it."
                return {"answer": text, "image": disp}
            if "altitude" in q:
                return f"{rec['callsign']} altitude: {rec.get('altitude', 'unknown')} ft."
            if "heading" in q or "direction" in q:
                return f"{rec['callsign']} heading: {rec.get('heading', 'unknown')} degrees."
            return f"{rec['callsign']} is detected in the {('top' if match['cy'] < 112 else 'bottom')}-{'left' if match['cx'] < 112 else 'right'} quadrant."

        # Relative box position questions can return the identity only when matched.
        positional_extreme = None
        if re.search(r"\b(leftmost|furthest left|far left)\b", q): positional_extreme = min
        elif re.search(r"\b(rightmost|furthest right|far right)\b", q): positional_extreme = max
        elif re.search(r"\b(topmost|highest on the image|uppermost)\b", q): positional_extreme = min
        elif re.search(r"\b(bottommost|lowest on the image|lowermost)\b", q): positional_extreme = max
        if positional_extreme:
            key = "cx" if re.search(r"left|right", q) else "cy"
            selected = positional_extreme(aircraft, key=lambda a: a[key])
            rec = selected["record"]
            if rec:
                return f"{rec['callsign']} is the {('leftmost' if key == 'cx' and selected['cx'] < 112 else 'rightmost') if key == 'cx' else ('topmost' if selected['cy'] < 112 else 'bottommost')} detected aircraft."
            return "I found the relevant YOLOv5 detection, but couldn't match it to a callsign in this frame."

        # Heading questions without a callsign ("heading of the aircraft nearest to the center"):
        # the CNN was trained on these, so let it answer instead of the catch-all below.
        if re.search(r"\bheading\b", q):
            return None

        # Callsign/metadata or general YOLO detection questions should not fall through to CNN's generic 'callsign' class.
        if any(word in q for word in ("callsign", "aircraft", "detected", "bounding box", "box around", "heading", "altitude", "highlight", "locate")):
            matched_count = sum(1 for ac in aircraft if ac["record"])
            return f"YOLOv5 detected {len(aircraft)} aircraft; {matched_count} were matched to callsign metadata in this frame. Please specify a callsign, position, altitude, heading, count, or quadrant comparison."
        return None

    def evaluate_and_answer(self, image_path, question):
        img = Image.open(image_path).convert("RGB")
        result = self.cnn_engine.answer(img, question)
        answer, confidence = result["answer"], result.get("confidence", 0.0)

        verdict, note = "confident", None

        if self.COUNT_QUESTION.search(question) and not self._ALTITUDE_COUNT_RE.search(question):
            predicted_n = self._extract_int(answer)
            if predicted_n is not None:
                yolo_n = self._yolo_count(image_path)
                if abs(predicted_n - yolo_n) > self.count_tolerance:
                    verdict = "yolo_mismatch"
                    note = f"YOLO detected {yolo_n} aircraft, model predicted {predicted_n}"

        if verdict == "confident":
            if confidence < self.decline_threshold:
                verdict = "low_confidence_decline"
            elif confidence < self.hedge_threshold:
                verdict = "low_confidence_hedge"

        if verdict == "low_confidence_decline":
            final_answer = ("I'm not confident enough in an answer to that one -- "
                             "it's been flagged for review rather than guessed.")
            self._log_jsonl(self.escalation_log, {
                "question": question, "image_path": image_path,
                "model_answer": answer, "confidence": confidence,
                "reason": verdict, "note": note,
            })
        elif verdict == "yolo_mismatch":
            final_answer = f"{answer} -- but I'm not fully sure ({note}); flagged for review."
            self._log_jsonl(self.escalation_log, {
                "question": question, "image_path": image_path,
                "model_answer": answer, "confidence": confidence,
                "reason": verdict, "note": note,
            })
        elif verdict == "low_confidence_hedge":
            final_answer = f"{answer} -- I think, though I'm only moderately confident."
        else:
            final_answer = answer

        if verdict != "confident":
            self._log_jsonl(self.retrain_log, {
                "question": question, "image_path": image_path,
                "model_answer": answer, "confidence": confidence,
                "reason": verdict, "note": note,
            })

        return {"answer": final_answer, "raw_answer": answer,
                "confidence": confidence, "verdict": verdict, "note": note}

    # ---------------------------------------------------------------
    def handle_question(self, image_path, question):
        """Single entry point -- same role CopilotSystem.answer() plays for
        the cockpit tab. Always returns:
        {"kind": "irrelevant"|"locate"|"vqa", "answer": str,
         "image": PIL.Image|None, "confidence": float|None, "verdict": str|None}
        """
        if not self.classify_relevance(question):
            return {
                "kind": "irrelevant",
                "answer": ("That's not something I can answer from this radar "
                           "image -- try asking about aircraft count, position, "
                           "altitude, heading, or a callsign."),
                "image": None, "confidence": None, "verdict": None,
            }

        # Route detection/callsign/attribute questions through YOLOv5 and same-frame metadata first.
        yolo_answer = self._answer_yolo_question(image_path, question)

        if yolo_answer is not None:
            from_boxes = getattr(self, "_last_answer_source", "yolo") == "yolo"
            detector_confidence = getattr(self, "_last_yolo_confidence", None) if from_boxes else None

            if isinstance(yolo_answer, dict):
                return {
                    "kind": "locate",
                    "answer": yolo_answer["answer"],
                    "image": yolo_answer.get("image"),
                    "confidence": detector_confidence,
                    "verdict": "yolo_metadata_match",
                    "confidence_type": "detector",
                }

            return {
                "kind": "vqa",
                "answer": yolo_answer,
                "image": None,
                "confidence": detector_confidence,
                # yolo_count = computed from YOLO boxes; yolo_metadata_match = read from frame metadata
                "verdict": "yolo_count" if from_boxes else "yolo_metadata_match",
                "confidence_type": "detector",
            }
        # Answer highest/lowest-altitude callsign questions from the selected
        # frame's ground-truth metadata, not the CNN's generic answer class.
        altitude_callsign_question = bool(
            re.search(r"\bcallsign\b", question, re.IGNORECASE)
            and re.search(r"\baltitude\b", question, re.IGNORECASE)
            and re.search(r"\b(highest|lowest|most|least)\b", question, re.IGNORECASE)
        )
        if altitude_callsign_question:
            answer = self._answer_altitude_extreme_callsign(image_path, question)
            return {
                "kind": "vqa",
                "answer": answer,
                "image": None,
                "confidence": None,
                "verdict": "metadata_altitude_extreme",
            }

        # Handle quadrant comparisons using detector coordinates rather than
        # relying on the CNN's answer-class confidence for this task.
        quadrant_question = bool(re.search(
            r"\b(quadrant|which part|which area).*\b(most|more|highest)\b"
            r"|\b(most|more|highest).*\baircraft\b.*\b(quadrant|area|part)\b",
            question,
            re.IGNORECASE,
        ))
        if quadrant_question:
            return {
                "kind": "vqa",
                "answer": self._answer_quadrant(image_path),
                "image": None,
                "confidence": None,
                "verdict": "yolo_quadrant_count",
            }

        # Route callsign-related questions to the locator even if they do not
        # literally start with "find", "locate", or "highlight".
        callsign = self.extract_callsign(question)
        locate_intent = bool(self.LOCATE_TRIGGER.search(question))
        callsign_intent = bool(
            callsign and re.search(
                r"\b(callsign|flight|aircraft|position|location|where|"
                r"locate|find|highlight|show)\b",
                question,
                re.IGNORECASE,
            )
        )

        if locate_intent or callsign_intent:
            result = self.locate_and_highlight(question)
            if result is not None:
                return {
                    "kind": "locate",
                    "answer": result["answer"],
                    "image": result.get("image"),
                    "confidence": None,
                    "verdict": None,
                }

        result = self.evaluate_and_answer(image_path, question)
        return {"kind": "vqa", "answer": result["answer"], "image": None,
                "confidence": result["confidence"], "verdict": result["verdict"]}