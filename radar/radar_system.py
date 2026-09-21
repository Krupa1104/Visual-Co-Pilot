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
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x):
        return self.proj(self.features(x))


class _LegacyQuestionEncoder(nn.Module):
    def __init__(self, vocab=40, embed=64, hidden=256, out_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab + 1, embed, padding_idx=0)
        self.gru = nn.GRU(embed, hidden, num_layers=2, batch_first=True,
                           dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, out_dim),
            nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(0.15),
        )

    def forward(self, x):
        emb = self.embed(x)
        _, h = self.gru(emb)
        return self.proj(torch.cat([h[-2], h[-1]], dim=-1))


class _LegacyCNNVQAModel(nn.Module):
    def __init__(self, num_classes, device="cpu", feat_dim=512):
        super().__init__()
        self.device = device
        self.visual_enc = _LegacyCNNVisualEncoder(out_dim=feat_dim)
        self.question_enc = _LegacyQuestionEncoder(out_dim=feat_dim)
        self.cross_attn = nn.MultiheadAttention(feat_dim, num_heads=8,
                                                 dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(feat_dim, feat_dim // 2), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(feat_dim // 2, num_classes),
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
        fused = self.norm1(v_s + att).squeeze(1)
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
    def _load_yolo(self, yolo_conf):
        yolo_weights = os.path.join(self.base_dir, "models", "yolov5_aircraft", "weights", "best.pt")
        if not os.path.exists(yolo_weights):
            raise FileNotFoundError(
                f"Fine-tuned YOLOv5 weights not found at {yolo_weights}. "
                "Copy best.pt from the teammate's Drive "
                "(models/yolov5_aircraft/weights/best.pt).")
        if not os.path.exists(self.yolo_root):
            raise FileNotFoundError(
                f"YOLOv5 repo not found at {self.yolo_root}. Run "
                f"`git clone https://github.com/ultralytics/yolov5 {self.yolo_root}` "
                f"and `pip install -r {self.yolo_root}/requirements.txt` once.")

        # Your repo has its OWN top-level `models` package (cnn_model.py
        # etc.) which collides with YOLOv5's own `models/common.py`.
        # Temporarily clear it out of sys.modules/sys.path while torch.hub
        # loads YOLOv5, then put YOUR `models` package back so anything
        # constructed after this point still gets the right one.
        own_models_mod = sys.modules.get("models")
        for mod_name in list(sys.modules):
            if mod_name == "models" or mod_name.startswith("models."):
                del sys.modules[mod_name]
        if self.yolo_root in sys.path:
            sys.path.remove(self.yolo_root)
        sys.path.insert(0, self.yolo_root)

        # best.pt was saved on Linux/Colab and has a PosixPath object
        # pickled inside its checkpoint. Windows' pathlib can't instantiate
        # PosixPath (it only has WindowsPath) -- this temporarily aliases it
        # so unpickling succeeds, then restores it right after.
        _real_posix_path = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        try:
            self.yolo_model = torch.hub.load(self.yolo_root, "custom", path=yolo_weights, source="local")
        finally:
            pathlib.PosixPath = _real_posix_path
        self.yolo_model.conf = yolo_conf

        sys.path.remove(self.yolo_root)
        for mod_name in list(sys.modules):
            if mod_name == "models" or mod_name.startswith("models."):
                del sys.modules[mod_name]
        if own_models_mod is not None:
            sys.modules["models"] = own_models_mod

    # ---------------------------------------------------------------
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
            cx, cy = (row.xmin + row.xmax) / 2, (row.ymin + row.ymax) / 2
            d = (cx - target_x) ** 2 + (cy - target_y) ** 2
            if d < best_dist:
                best_dist, matched = d, row

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

    def evaluate_and_answer(self, image_path, question):
        img = Image.open(image_path).convert("RGB")
        result = self.cnn_engine.answer(img, question)
        answer, confidence = result["answer"], result.get("confidence", 0.0)

        verdict, note = "confident", None

        if self.COUNT_QUESTION.search(question):
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

        if self.LOCATE_TRIGGER.search(question):
            result = self.locate_and_highlight(question)
            if result is not None:
                return {"kind": "locate", "answer": result["answer"],
                        "image": result["image"], "confidence": None, "verdict": None}

        result = self.evaluate_and_answer(image_path, question)
        return {"kind": "vqa", "answer": result["answer"], "image": None,
                "confidence": result["confidence"], "verdict": result["verdict"]}