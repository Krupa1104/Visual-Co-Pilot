"""
Radar inference system for radar2.

Expected layout:
radar2/
    radar_system.py
    models/yolov5/weights/best.pt
    models/vqa/best.pt
    src/models/radar_vqa_model.py
        configs/answer_mapping.py
    data/qa_complete/annotations.jsonl
    data/synthetic_complete/

Uses YOLOv5 for aircraft detection, RadarVQAModel for VQA,
and PaddleOCR 3.x for visible callsign/altitude/heading text.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
sys.modules.setdefault("pathlib._local", pathlib)

CALLSIGN_PATTERN = re.compile(
    r"\b[A-Z]{2,5}\d{1,5}[A-Z]?\b", re.IGNORECASE
)
COUNT_QUESTION = re.compile(
    r"\b(how many|number of|count)\b", re.IGNORECASE
)

RELEVANCE_TERMS = {
    "aircraft", "airplane", "plane", "flight", "radar",
    "callsign", "call sign", "altitude", "heading",
    "airborne", "aviation", "helicopter", "pilot",
    "runway", "sky", "detected", "flying",
}

# Radar scope centre (the green dot) as fractions of image width/height. Measured 4 Nov 2026 from the
# dot in radar_02232.png (1280x1280): (640, 655), i.e. 15 px BELOW the image centre. The dataset's
# quadrant labels (north-east, ...) are compass quadrants around this point.
SCOPE_CENTRE_FRAC = (640 / 1280, 655 / 1280)
COMPASS8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
COMPASS_WORDS = {"north": "N", "north-east": "NE", "northeast": "NE", "east": "E",
                 "south-east": "SE", "southeast": "SE", "south": "S", "south-west": "SW",
                 "southwest": "SW", "west": "W", "north-west": "NW", "northwest": "NW"}
HEADING_YESNO = re.compile(
    r"^\s*(is|are)\s+(there\s+)?any\s+(aircraft|airplanes?|planes?|flights?)\s+(heading|flying|going)"
    r"\s+(towards?\s+|to\s+(the\s+)?)?(?P<dir>north-?east|north-?west|south-?east|south-?west|north|south|"
    r"east|west|NE|NW|SE|SW|N|S|E|W)\b", re.IGNORECASE)
COMPASS_QUADRANT = re.compile(r"\b(north|south)-?(east|west)\b", re.IGNORECASE)


def heading_to_compass8(degrees):
    return COMPASS8[int((float(degrees) % 360 + 22.5) // 45) % 8]


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


class RadarSystem:
    """Inference facade used by integrated.py."""
    EXCLUDE_TOP = 0.08
    EXCLUDE_BOTTOM = 0.07
    def __init__(
        self,
        base_dir: str,
        repo_root: Optional[str] = None,
        yolo_root: Optional[str] = None,
        yolo_conf: float = 0.25,
        device: Optional[str] = None,
        llm_tokenizer: Any = None,
        llm_model: Any = None,
        hedge_threshold: Optional[float] = None,
        decline_threshold: Optional[float] = None,
        count_tolerance: Optional[int] = None,
        cnn_checkpoint: Optional[str] = None,
        vqa_checkpoint: Optional[str] = None,
        yolo_checkpoint: Optional[str] = None,
        answer_index_path: Optional[str] = None,
        max_question_length: int = 60,
        enable_ocr: bool = True,
        ocr_models: str = "mobile",
        ocr_mode: str = "crops",
        ocr_crop_pad_x: float = 4.0,
        ocr_crop_pad_y: float = 1.5,
        # fraction treated as footer text
        **_unused_kwargs: Any,
    ) -> None:

        self.base_dir = os.path.abspath(base_dir)
        self.repo_root = os.path.abspath(repo_root or self.base_dir)

        self.yolo_root = os.path.abspath(
            yolo_root or os.path.join(
                self.base_dir, "models", "yolov5"
            )
        )

        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.yolo_conf = float(yolo_conf)

        self.hedge_threshold = float(
            hedge_threshold if hedge_threshold is not None else 0.55
        )
        self.decline_threshold = float(
            decline_threshold if decline_threshold is not None else 0.30
        )
        self.count_tolerance = int(
            count_tolerance if count_tolerance is not None else 1
        )

        self.max_question_length = int(max_question_length)
        self.enable_ocr = bool(enable_ocr)
        # OCR speed settings (added 4 Nov 2026, Krupa; measured with scripts/ocr_speed_check.py and
        # scripts/ocr_crop_check.py on 10 images: default v6 medium models on the full image 28.9 s/image,
        # v5 mobile models 8.1 s/image, mobile models on crops around YOLO aircraft 5.2 s/image, all 19/20
        # callsigns read). The original behaviour is ocr_models="medium", ocr_mode="full".
        #   ocr_models: "mobile" = PP-OCRv5_mobile det+rec, "medium" = PaddleOCR's default models
        #   ocr_mode:   "crops"  = OCR only the area around each detected aircraft (its label is drawn
        #                          next to it); "full" = whole image, then YOLO text-box crops if incomplete
        if ocr_models not in ("mobile", "medium") or ocr_mode not in ("crops", "full"):
            raise ValueError("ocr_models must be 'mobile'/'medium', ocr_mode 'crops'/'full'")
        self.ocr_models = ocr_models
        self.ocr_mode = ocr_mode
        self.ocr_crop_pad_x = float(ocr_crop_pad_x)
        self.ocr_crop_pad_y = float(ocr_crop_pad_y)

        self.llm_tokenizer = llm_tokenizer
        self.llm_model = llm_model

        self._ocr_reader = None
        self._last_ocr_confidence = None
        self._last_ocr_items = []

        self.logs_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        self.logger = logging.getLogger(
            f"RadarSystem:{self.base_dir}"
        )
        self.logger.setLevel(logging.INFO)

        self.vqa_checkpoint = os.path.abspath(
            vqa_checkpoint or os.path.join(
                self.base_dir, "models", "vqa", "best.pt"
            )
        )

        self.yolo_checkpoint = os.path.abspath(
            yolo_checkpoint or os.path.join(
                self.base_dir, "models", "yolov5",
                "weights", "best.pt"
            )
        )

        

        # Make radar2 importable for src.models.radar_vqa_model.
        # Make radar2 importable for its model and configuration modules.
        if self.base_dir not in sys.path:
            sys.path.insert(0, self.base_dir)

        from configs.answer_mapping import (
            ANSWER_TO_ID,
            ID_TO_ANSWER,
            NUM_OUTPUTS,
            NUM_VALID_CLASSES,
            RESERVED_CLASS_ID,
            get_valid_logits,
            decode_answer,
        )
        self.get_valid_logits = get_valid_logits
        self.decode_answer = decode_answer

        if NUM_OUTPUTS != 28 or NUM_VALID_CLASSES != 27:
            raise ValueError("Unexpected Radar VQA answer mapping configuration.")

        if RESERVED_CLASS_ID != 27:
            raise ValueError("Expected reserved classifier class ID 27.")

        self.vqa_model = self._load_vqa_model()

# Use the validated shared answer mapping.
        self.answer_to_index = ANSWER_TO_ID
        self.index_to_answer = ID_TO_ANSWER

        self.yolo_model = self._load_yolo_model()

        if self.enable_ocr:
            self._load_ocr()

    # ------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------

    def _load_vqa_model(self):

        model_file = os.path.join(
            self.base_dir, "src", "models",
            "radar_vqa_model.py"
        )

        if not os.path.isfile(model_file):
            raise FileNotFoundError(
                f"VQA architecture not found: {model_file}"
            )

        if not os.path.isfile(self.vqa_checkpoint):
            raise FileNotFoundError(
                f"VQA checkpoint not found: {self.vqa_checkpoint}"
            )

        try:
            from src.models.radar_vqa_model import RadarVQAModel

        except Exception as exc:
            raise ImportError(
                "Could not import RadarVQAModel from "
                "radar2/src/models/radar_vqa_model.py. "
                "Check that src and models are importable packages."
            ) from exc

        # Avoid downloading ImageNet weights. The trained checkpoint
        # should contain the complete model state.
        model = RadarVQAModel(pretrained=False)

        try:
            checkpoint = torch.load(
                self.vqa_checkpoint,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(
                self.vqa_checkpoint,
                map_location=self.device
            )

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            elif (
                "model" in checkpoint
                and isinstance(checkpoint["model"], dict)
            ):
                state_dict = checkpoint["model"]

            else:
                state_dict = checkpoint

        else:
            raise RuntimeError(
                "Unsupported VQA checkpoint format. Expected a "
                "state_dict or a dictionary containing "
                "model_state_dict/state_dict."
            )

        # Handle DataParallel checkpoints.
        if state_dict and all(
            str(key).startswith("module.")
            for key in state_dict
        ):
            state_dict = {
                str(key)[7:]: value
                for key, value in state_dict.items()
            }

        model.load_state_dict(state_dict, strict=True)

        model.to(self.device)
        model.eval()

        return model

    

    def _load_yolo_model(self):

        if not os.path.isdir(self.yolo_root):
            raise FileNotFoundError(
                f"Local YOLOv5 repository not found: {self.yolo_root}"
            )

        if not os.path.isfile(self.yolo_checkpoint):
            raise FileNotFoundError(
                f"YOLO weights not found: {self.yolo_checkpoint}"
            )

        # Windows compatibility for checkpoints saved on Linux.
        original_posix_path = pathlib.PosixPath

        try:
            pathlib.PosixPath = pathlib.WindowsPath

            model = torch.hub.load(
                self.yolo_root,
                "custom",
                path=self.yolo_checkpoint,
                source="local",
                force_reload=False,
            )

        finally:
            pathlib.PosixPath = original_posix_path

        model.to(self.device)
        model.conf = self.yolo_conf

        if hasattr(model, "iou"):
            model.iou = 0.45

        return model

    def _load_ocr(self):

        try:
            from paddleocr import PaddleOCR

            # enable_mkldnn stays False: True fails with "ConvertPirAttribute2RuntimeAttribute not
            # support" (radar team, and re-checked 4 Nov with scripts/ocr_speed_check.py).
            if self.ocr_models == "mobile":
                model_args = dict(text_detection_model_name="PP-OCRv5_mobile_det",
                                  text_recognition_model_name="PP-OCRv5_mobile_rec")
            else:
                model_args = dict(lang="en")
            self._ocr_reader = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
                **model_args,
            )

        except Exception as exc:
            self.logger.exception(
                "PaddleOCR initialization failed"
            )

            raise RuntimeError(
                "PaddleOCR 3.x could not be initialized. "
                "Verify paddleocr and paddlepaddle installations."
            ) from exc

    # ------------------------------------------------------------
    # Relevance and input preparation
    # ------------------------------------------------------------

    def _is_radar_question(self, question: str) -> bool:

        if not question or not question.strip():
            return False

        # Use the shared LLM when integrated.py provides one.
        if (
            self.llm_model is not None
            and self.llm_tokenizer is not None
        ):
            try:
                prompt = (
                    "Decide whether the user's question can be "
                    "answered from an image of an aircraft radar "
                    "display. Reply with only YES or NO.\n"
                    f"Question: {question}"
                )

                inputs = self.llm_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                )

                model_device = next(
                    self.llm_model.parameters()
                ).device

                inputs = {
                    key: value.to(model_device)
                    for key, value in inputs.items()
                }

                with torch.inference_mode():
                    output = self.llm_model.generate(
                        **inputs,
                        max_new_tokens=3,
                        do_sample=False,
                    )

                decoded = self.llm_tokenizer.decode(
                    output[0][
                        inputs["input_ids"].shape[1]:
                    ],
                    skip_special_tokens=True,
                ).strip().upper()

                if decoded.startswith("YES"):
                    return True

                if decoded.startswith("NO"):
                    return False

            except Exception:
                self.logger.exception(
                    "Shared LLM relevance check failed"
                )

        lowered = question.lower()

        return any(
            term in lowered for term in RELEVANCE_TERMS
        )

    def _prepare_image_tensor(self, image_path: str):

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = IMAGE_TRANSFORM(image).unsqueeze(0)

        return tensor.to(self.device)

    def _prepare_question_tensor(self, question: str):

        from src.models.radar_vqa_model import encode_text, PAD_ID

        ids = encode_text(question)
        ids = ids[:self.max_question_length]

        if not ids:
            ids = [37]  # UNK_ID

        actual_length = len(ids)

        padded = ids + [
            PAD_ID
        ] * (self.max_question_length - len(ids))

        question_tensor = torch.tensor(
            [padded],
            dtype=torch.long,
            device=self.device,
        )

        length_tensor = torch.tensor(
            [actual_length],
            dtype=torch.long,
            device=self.device,
        )

        return question_tensor, length_tensor

    # ------------------------------------------------------------
    # YOLO aircraft detection
    # ------------------------------------------------------------

    
    
    def _detect_aircraft(self, image_path: str):
        results = self.yolo_model(image_path, size=1280)

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        height, width = frame.shape[:2]

        aircraft_detections = []
        text_detections = []

        AIRCRAFT_CONFIDENCE_THRESHOLD = 0.55

        # Obtain detection rows from YOLO.
        try:
            raw = results.pandas().xyxy[0]

            print("\n========== RAW YOLO RESULTS ==========", flush=True)
            print(f"Total raw detections: {len(raw)}", flush=True)

            if not raw.empty:
                print(
                    raw[[
                        "name", "class", "confidence",
                        "xmin", "ymin", "xmax", "ymax"
                    ]].to_string(index=False),
                    flush=True,
                )

            rows = raw.to_dict("records")

        except Exception:
            print("\n========== RAW YOLO RESULTS ==========", flush=True)

            try:
                prediction = results.xyxy[0].detach().cpu().numpy()
                names = getattr(self.yolo_model, "names", {})
                rows = []

                for item in prediction:
                    x1, y1, x2, y2, confidence, class_value = item[:6]
                    class_id = int(class_value)

                    if isinstance(names, dict):
                        class_name = names.get(class_id, str(class_id))
                    elif isinstance(names, (list, tuple)) and class_id < len(names):
                        class_name = names[class_id]
                    else:
                        class_name = str(class_id)

                    rows.append({
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                        "confidence": confidence,
                        "class": class_id,
                        "name": class_name,
                    })

                print(f"Total raw detections: {len(rows)}", flush=True)
                for row in rows:
                    print(row, flush=True)

            except Exception as exc:
                raise RuntimeError(
                    "Could not parse YOLOv5 detection results."
                ) from exc

        print("=======================================\n", flush=True)

        # Class 0 = aircraft; class 1 = callsign_text.
        for row in rows:
            class_id = int(row.get("class", -1))

            if class_id not in (0, 1):
                continue

            confidence = float(row.get("confidence", 0.0))

            x1 = max(0, min(width, int(float(row["xmin"]))))
            y1 = max(0, min(height, int(float(row["ymin"]))))
            x2 = max(0, min(width, int(float(row["xmax"]))))
            y2 = max(0, min(height, int(float(row["ymax"]))))

            if x2 <= x1 or y2 <= y1:
                continue

            detection = {
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_id": class_id,
                "name": str(
                    row.get(
                        "name",
                        "aircraft" if class_id == 0 else "callsign_text"
                    )
                ),
            }

            if class_id == 0:
                # Apply the aircraft threshold only to class 0, and ignore boxes
                # centred in the header/footer text bands (the model mistakes
                # that text for aircraft at ~0.56 confidence).
                centre_y = (y1 + y2) / 2.0
                in_scope = (self.EXCLUDE_TOP * height
                            <= centre_y <=
                            (1.0 - self.EXCLUDE_BOTTOM) * height)
                if confidence >= AIRCRAFT_CONFIDENCE_THRESHOLD and in_scope:
                    aircraft_detections.append(detection)

            else:
                # Preserve text detections independently of aircraft threshold.
                text_detections.append(detection)

        # Remove overlapping duplicate aircraft boxes using IoU.
        def calculate_iou(box_a, box_b):
            ax1, ay1, ax2, ay2 = box_a
            bx1, by1, bx2, by2 = box_b

            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)

            intersection = (
                max(0, ix2 - ix1) * max(0, iy2 - iy1)
            )

            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

            union = area_a + area_b - intersection

            return intersection / union if union > 0 else 0.0

        candidates = sorted(
            aircraft_detections,
            key=lambda item: item["confidence"],
            reverse=True,
        )

        filtered_aircraft = []

        for candidate in candidates:
            is_duplicate = any(
                calculate_iou(candidate["bbox"], accepted["bbox"]) >= 0.50
                for accepted in filtered_aircraft
            )

            if not is_duplicate:
                filtered_aircraft.append(candidate)

        self._last_text_detections = sorted(
            text_detections,
            key=lambda item: item["confidence"],
            reverse=True,
        )

        print(
            f"[RADAR] Aircraft above confidence threshold "
            f"({AIRCRAFT_CONFIDENCE_THRESHOLD:.2f}): "
            f"{len(aircraft_detections)}",
            flush=True,
        )

        print(
            f"[RADAR] Aircraft after duplicate filtering: "
            f"{len(filtered_aircraft)}",
            flush=True,
        )

        print(
            f"[RADAR] YOLO text-box fallback candidates: "
            f"{len(self._last_text_detections)}",
            flush=True,
        )

        return filtered_aircraft



    @staticmethod
    def _mean_detector_confidence(detections):

        values = [
            detection["confidence"]
            for detection in detections
            if detection.get("confidence") is not None
        ]

        return (
            float(np.mean(values))
            if values else None
        )

    def _annotate(
        self,
        image_path: str,
        detections: Sequence[Dict[str, Any]],
        labels: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        for index, detection in enumerate(
            detections, start=1
        ):
            x1, y1, x2, y2 = detection["bbox"]

            draw.rectangle(
                (x1, y1, x2, y2),
                outline=(0, 220, 255),
                width=3,
            )

            draw.text(
                (x1, max(0, y1 - 22)),
                f"Aircraft {index} ({detection['confidence']:.2f})",
                fill=(0, 220, 255),
                font=font,
            )

        if labels:
            for item in labels:
                x, y = item.get("center", (0, 0))

                raw_text = item.get("text", "")
                if isinstance(raw_text, (list, tuple)):
                    text = " ".join(str(part) for part in raw_text)
                else:
                    text = str(raw_text)

                text = text.strip()
                if not text:
                    continue

                draw.text(
                    (int(x), int(y)),
                    text,
                    fill=(255, 220, 0),
                    font=font,
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )
        output_dir = os.path.join(
            self.base_dir, "outputs", "annotated"
        )
        os.makedirs(output_dir, exist_ok=True)

        stem = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        output_path = os.path.join(
            output_dir,
            f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg",
        )

        image.save(output_path, quality=95)

        return output_path

    # ------------------------------------------------------------
    # Geometry-based YOLO answers
    # ------------------------------------------------------------

    @staticmethod
    def _quadrant_for_box(bbox, width: int, height: int):
        """top-left / top-right / bottom-left / bottom-right around the radar SCOPE centre
        (was the image centre until 4 Nov 2026; the scope centre is 15 px lower)."""

        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        ox, oy = SCOPE_CENTRE_FRAC[0] * width, SCOPE_CENTRE_FRAC[1] * height

        horizontal = (
            "left" if cx < ox else "right"
        )
        vertical = (
            "top" if cy < oy else "bottom"
        )

        return f"{vertical}-{horizontal}"

    @staticmethod
    def _compass_quadrant(screen_quadrant):
        return {"top-left": "north-west", "top-right": "north-east",
                "bottom-left": "south-west", "bottom-right": "south-east"}[screen_quadrant]

    def _answer_yolo(
        self,
        image_path: str,
        question: str,
        detections: Sequence[Dict[str, Any]],
    ) -> Optional[str]:

        lowered = question.lower()

        with Image.open(image_path) as image:
            width, height = image.size

        count = len(detections)

        if (
            COUNT_QUESTION.search(lowered)
            and any(
                word in lowered
                for word in (
                    "aircraft", "airplane", "plane",
                    "flight", "object"
                )
            )
        ):
            return (
                f"I detect {count} aircraft in the radar image."
            )

        quadrant_terms = {
            "top-left": (
                "top left", "upper left", "top-left"
            ),
            "top-right": (
                "top right", "upper right", "top-right"
            ),
            "bottom-left": (
                "bottom left", "lower left", "bottom-left"
            ),
            "bottom-right": (
                "bottom right", "lower right", "bottom-right"
            ),
        }

        for quadrant, phrases in quadrant_terms.items():

            if any(
                phrase in lowered for phrase in phrases
            ):
                number = sum(
                    self._quadrant_for_box(
                        detection["bbox"], width, height
                    ) == quadrant
                    for detection in detections
                )

                return (
                    f"I detect {number} aircraft in the "
                    f"{quadrant.replace('-', ' ')} quadrant."
                )

        if (
            "quadrant" in lowered
            and ("most" in lowered or "highest" in lowered)
        ):
            counts = {
                quadrant: 0
                for quadrant in (
                    "top-left", "top-right",
                    "bottom-left", "bottom-right"
                )
            }

            for detection in detections:
                quadrant = self._quadrant_for_box(
                    detection["bbox"], width, height
                )
                counts[quadrant] += 1

            maximum = max(counts.values(), default=0)

            if maximum == 0:
                return (
                    "I do not detect any aircraft in the image."
                )

            winners = [
                quadrant
                for quadrant, value in counts.items()
                if value == maximum
            ]

            if len(winners) > 1:
                return (
                    f"The highest count is tied between "
                    f"{', '.join(winners)} with {maximum} "
                    "aircraft each."
                )

            return (
                f"The {winners[0].replace('-', ' ')} quadrant "
                f"has the most aircraft ({maximum})."
            )

        if any(
            word in lowered
            for word in (
                "left", "right", "top", "bottom",
                "center", "centre"
            )
        ):

            asks_presence = any(
                word in lowered
                for word in (
                    "any", "is there", "are there",
                    "present", "located"
                )
            )

            if asks_presence:

                region = (
                    "center"
                    if "center" in lowered or "centre" in lowered
                    else None
                )

                selected = []

                for detection in detections:

                    x1, y1, x2, y2 = detection["bbox"]

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    if region:
                        inside = (
                            width * 0.35 <= cx <= width * 0.65
                            and height * 0.35 <= cy <= height * 0.65
                        )
                    else:
                        inside = (
                            ("left" in lowered and cx < width / 2)
                            or (
                                "right" in lowered
                                and cx >= width / 2
                            )
                            or (
                                "top" in lowered
                                and cy < height / 2
                            )
                            or (
                                "bottom" in lowered
                                and cy >= height / 2
                            )
                        )

                    if inside:
                        selected.append(detection)

                location = region or next(
                    (
                        word for word in (
                            "left", "right", "top", "bottom"
                        )
                        if word in lowered
                    ),
                    "requested region",
                )

                if selected:
                    return (
                        f"Yes, I detect {len(selected)} aircraft "
                        f"in the {location} region."
                    )

                return (
                    f"No aircraft were detected in the "
                    f"{location} region."
                )

        return None
        # ------------------------------------------------------------
    # PaddleOCR 3.x text recognition
    # ------------------------------------------------------------

    @staticmethod
        
    def _extract_paddle_result(result):
        """Convert PaddleOCR 3.x output into normalized text rows."""

        payload = getattr(result, "json", result)

        if callable(payload):
            payload = payload()

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (ValueError, TypeError):
                return []

        if not isinstance(payload, dict):
            return []

        res = payload.get("res", payload)

        if not isinstance(res, dict):
            return []

        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        polygons = res.get("rec_polys")
        boxes = res.get("rec_boxes")

        # Convert array-like outputs into ordinary Python lists.
        if polygons is not None and hasattr(polygons, "tolist"):
            polygons = polygons.tolist()

        if boxes is not None and hasattr(boxes, "tolist"):
            boxes = boxes.tolist()

        rows = []

        for index, text in enumerate(texts):
            text = str(text).strip()

            if not text:
                continue

            try:
                score = float(scores[index]) if index < len(scores) else 0.0
            except (TypeError, ValueError):
                score = 0.0

            polygon = None
            center = None

            # Preferred: PaddleOCR polygon with four corner points.
            if polygons is not None and index < len(polygons):
                candidate = polygons[index]

                try:
                    points = [
                        [float(point[0]), float(point[1])]
                        for point in candidate
                        if len(point) >= 2
                    ]

                    if len(points) >= 2:
                        polygon = points
                        center = (
                            sum(point[0] for point in points) / len(points),
                            sum(point[1] for point in points) / len(points),
                        )
                except (TypeError, ValueError, IndexError):
                    polygon = None
                    center = None

            # Fallback: rectangle [x1, y1, x2, y2].
            if center is None and boxes is not None and index < len(boxes):
                try:
                    x1, y1, x2, y2 = map(float, boxes[index][:4])

                    if x2 >= x1 and y2 >= y1:
                        polygon = [
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2],
                        ]
                        center = ((x1 + x2) / 2, (y1 + y2) / 2)
                except (TypeError, ValueError, IndexError):
                    pass

            rows.append({
                "text": text,
                "confidence": score,
                "polygon": polygon,
                "center": center,
            })

        return rows

    @staticmethod
    def _merge_text_lines(rows):
        """Join OCR pieces that lie on the same text line into one row, left to right.
        The mobile OCR models return 'ALT' and '30400' (and 'HDG', '353') as separate pieces, which
        _parse_visible_label cannot read; the medium models return 'ALT 30400'. Pieces are joined
        when their vertical centres are within half a text height and the horizontal gap is less
        than 1.5 text heights (so two different labels side by side are not joined)."""
        boxes = []
        for r in rows:
            poly = r.get("polygon")
            if not poly:
                boxes.append((r, None))
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            boxes.append((r, (min(xs), min(ys), max(xs), max(ys))))
        merged = [r for r, b in boxes if b is None]
        items = sorted([(r, b) for r, b in boxes if b is not None], key=lambda t: (t[1][1] + t[1][3]) / 2)
        lines = []
        for r, b in items:
            cy, h = (b[1] + b[3]) / 2, b[3] - b[1]
            for line in lines:
                lcy, lh = line["cy"], line["h"]
                if abs(cy - lcy) <= 0.5 * max(h, lh):
                    line["parts"].append((r, b))
                    break
            else:
                lines.append({"cy": cy, "h": h, "parts": [(r, b)]})
        for line in lines:
            parts = sorted(line["parts"], key=lambda t: t[1][0])
            group = [parts[0]]
            for r, b in parts[1:]:
                prev_b = group[-1][1]
                if b[0] - prev_b[2] < 1.5 * max(line["h"], 1):
                    group.append((r, b))
                else:
                    merged.append(RadarSystem._join_parts(group))
                    group = [(r, b)]
            merged.append(RadarSystem._join_parts(group))
        return merged

    @staticmethod
    def _join_parts(group):
        if len(group) == 1:
            return group[0][0]
        x1 = min(b[0] for _, b in group); y1 = min(b[1] for _, b in group)
        x2 = max(b[2] for _, b in group); y2 = max(b[3] for _, b in group)
        return {"text": " ".join(str(r["text"]).strip() for r, _ in group),
                "confidence": min(float(r.get("confidence", 0.0)) for r, _ in group),
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "center": ((x1 + x2) / 2, (y1 + y2) / 2)}

    @staticmethod
    def _make_ocr_variants(crop):

        height, width = crop.shape[:2]

        if height == 0 or width == 0:
            return []

        scale = min(
            4.0,
            900.0 / max(height, width)
        )
        scale = max(1.0, scale)

        enlarged = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        variants = [enlarged]

        gray = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2GRAY,
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        enhanced = clahe.apply(gray)

        variants.append(
            cv2.cvtColor(
                enhanced,
                cv2.COLOR_GRAY2BGR,
            )
        )

        adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        variants.append(
            cv2.cvtColor(
                adaptive,
                cv2.COLOR_GRAY2BGR,
            )
        )

        inverse = cv2.bitwise_not(adaptive)

        variants.append(
            cv2.cvtColor(
                inverse,
                cv2.COLOR_GRAY2BGR,
            )
        )

        return variants

    
    
    
    def _ocr_text(self, image_path: str, detections=None):
        """
        Primary: run PaddleOCR on the complete uploaded radar image.
        Fallback: run PaddleOCR on YOLO class-1 text-region crops
        when full-image OCR does not find enough callsigns or altitudes.
        """
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Could not read image for OCR: {image_path}"
            )

        if self._ocr_reader is None:
            raise RuntimeError("PaddleOCR is not initialized.")

        def run_paddle_ocr(input_image):
            results = self._ocr_reader.predict(input_image)
            extracted = []

            for result in results or []:
                extracted.extend(
                    self._extract_paddle_result(result)
                )

            return extracted

        # Fast path (ocr_mode="crops"): OCR only a window around each detected aircraft, where its
        # label box is drawn, and translate the results back to full-image coordinates.
        if self.ocr_mode == "crops" and detections:
            height, width = image.shape[:2]
            ocr_items = []
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                bw, bh = x2 - x1, y2 - y1
                cx1 = max(0, int(x1 - self.ocr_crop_pad_x * bw))
                cx2 = min(width, int(x2 + self.ocr_crop_pad_x * bw))
                cy1 = max(0, int(y1 - self.ocr_crop_pad_y * bh))
                cy2 = min(height, int(y2 + self.ocr_crop_pad_y * bh))
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                for crop_item in self._merge_text_lines(run_paddle_ocr(image[cy1:cy2, cx1:cx2])):
                    item = dict(crop_item)
                    if not str(item.get("text", "")).strip():
                        continue
                    if item.get("polygon") is not None:
                        item["polygon"] = [[float(px) + cx1, float(py) + cy1]
                                           for px, py in item["polygon"]]
                    if item.get("center") is not None:
                        item["center"] = (float(item["center"][0]) + cx1,
                                          float(item["center"][1]) + cy1)
                    item["source"] = "aircraft_crop"
                    ocr_items.append(item)
            print(f"[OCR] Aircraft-crop PaddleOCR ({len(detections)} crops) extracted "
                  f"{len(ocr_items)} text items.", flush=True)
            self._last_ocr_items = ocr_items
            confidences = [float(i["confidence"]) for i in ocr_items if i.get("confidence") is not None]
            self._last_ocr_confidence = sum(confidences) / len(confidences) if confidences else None
            return ocr_items

        # 1. Primary OCR: process the complete uploaded image.
        ocr_items = run_paddle_ocr(image)

        ocr_items = [
            dict(item)
            for item in ocr_items
            if str(item.get("text", "")).strip()
        ]

        for item in ocr_items:
            item["source"] = "full_image_paddleocr"

        print(
            f"[OCR] Full-image PaddleOCR extracted "
            f"{len(ocr_items)} text items.",
            flush=True,
        )

        # 2. Count the unique fields recognized in the full image.
        recognized_callsigns = set()
        recognized_altitudes = set()

        for item in ocr_items:
            parsed = self._parse_visible_label(
                str(item.get("text", ""))
            )

            if parsed.get("callsign"):
                recognized_callsigns.add(parsed["callsign"])

            if parsed.get("altitude") is not None:
                recognized_altitudes.add(parsed["altitude"])

        expected_aircraft = len(detections or [])

        # Avoid treating an empty detection list as proof that OCR is incomplete.
        ocr_incomplete = (
            not ocr_items
            or (
                expected_aircraft > 0
                and (
                    len(recognized_callsigns) < expected_aircraft
                    or len(recognized_altitudes) < expected_aircraft
                )
            )
        )

        # 3. Fallback: process YOLO class-1 text crops.
        if ocr_incomplete:
            text_detections = getattr(
                self,
                "_last_text_detections",
                [],
            )

            if text_detections:
                print(
                    f"[OCR] Full-image OCR found "
                    f"{len(recognized_callsigns)} unique callsigns and "
                    f"{len(recognized_altitudes)} unique altitudes "
                    f"for {expected_aircraft} detected aircraft. "
                    f"Trying {len(text_detections)} YOLO text crops.",
                    flush=True,
                )

                height, width = image.shape[:2]

                for detection in text_detections:
                    x1, y1, x2, y2 = detection["bbox"]

                    x1 = max(0, min(width, int(x1)))
                    y1 = max(0, min(height, int(y1)))
                    x2 = max(0, min(width, int(x2)))
                    y2 = max(0, min(height, int(y2)))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = image[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    crop_items = run_paddle_ocr(crop)

                    for crop_item in crop_items:
                        item = dict(crop_item)

                        # Translate crop-local polygon coordinates
                        # into the original image coordinate system.
                        polygon = item.get("polygon")

                        if polygon is not None:
                            item["polygon"] = [
                                [
                                    float(point[0]) + x1,
                                    float(point[1]) + y1,
                                ]
                                for point in polygon
                            ]

                        center = item.get("center")

                        if center is not None:
                            item["center"] = (
                                float(center[0]) + x1,
                                float(center[1]) + y1,
                            )

                        item["source"] = "yolo_crop_fallback"
                        ocr_items.append(item)

                print(
                    f"[OCR] Total text items after fallback: "
                    f"{len(ocr_items)}",
                    flush=True,
                )
            else:
                print(
                    "[OCR] Full-image OCR was incomplete, "
                    "but no YOLO text crops were available.",
                    flush=True,
                )

        # 4. Save OCR results for downstream association.
        self._last_ocr_items = ocr_items

        confidences = [
            float(item["confidence"])
            for item in ocr_items
            if item.get("confidence") is not None
        ]

        self._last_ocr_confidence = (
            sum(confidences) / len(confidences)
            if confidences else None
        )

        return ocr_items

    @staticmethod
    def _parse_visible_label(text):

        upper = text.upper().replace(",", " ")
        # ICAO airline callsigns start with 3 LETTERS; OCR sometimes reads O as 0 (IGO452 -> IG0452).
        # Fix look-alike digits in the first 3 characters of a 3-char + 2-4-digit token (4 Nov 2026).
        def _fix(match):
            head = match.group(1).translate(str.maketrans("0158", "OISB"))
            return head + match.group(2) if head.isalpha() else match.group(0)
        upper = re.sub(r"\b([A-Z0-9]{3})(\d{2,4})\b",
                       lambda mm: _fix(mm) if sum(ch.isalpha() for ch in mm.group(1)) >= 2 else mm.group(0),
                       upper)

        callsign_match = CALLSIGN_PATTERN.search(
            upper
        )

        callsign = (
            callsign_match.group(0)
            if callsign_match
            else None
        )

        altitude = None

        altitude_match = re.search(
            r"\b(?:ALT(?:ITUDE)?\s*[:=]?\s*)?"
            r"(\d{2,3}(?:,\d{3})?|\d{4,6})"
            r"\s*(?:FT|FEET)\b",
            upper,
        )

        if altitude_match:

            altitude = int(
                altitude_match.group(1).replace(",", "")
            )

        else:

            altitude_match = re.search(
                r"\bALT(?:ITUDE)?\s*[:=]?\s*(\d{3,6})\b",
                upper,
            )

            if altitude_match:
                altitude = int(
                    altitude_match.group(1)
                )

        heading = None

        heading_match = re.search(
            r"\b(?:HDG|HEADING)\s*[:=]?\s*(\d{1,3})\b",
            upper,
        )

        if heading_match:
            heading = int(
                heading_match.group(1)
            ) % 360

        return {
            "callsign": callsign,
            "altitude": altitude,
            "heading": heading,
        }

    
    def _associate_ocr(
        self,
        detections,
        ocr_items,
        image_size,
    ):
        width, height = image_size

        # One combined record per detected aircraft.
        associated_by_aircraft = {}

        # Keep readable text that could not be assigned separately.
        unassociated = []

        for index in range(len(detections)):
            associated_by_aircraft[index] = {
                "callsign": None,
                "altitude": None,
                "heading": None,
                "confidence": 0.0,
                "aircraft_index": index,
                "text": [],
                "center": None,
            }

        for item in ocr_items:
            text = str(item.get("text", "")).strip()

            if not text:
                continue

            parsed = self._parse_visible_label(text)

            if not any(parsed.values()):
                continue

            center = item.get("center")

            if center is None or len(center) < 2:
                continue

            cx = float(center[0])
            cy = float(center[1])

            confidence = float(item.get("confidence", 0.0))

            candidates = []

            for index, detection in enumerate(detections):
                x1, y1, x2, y2 = detection["bbox"]

                nearest_x = min(max(cx, x1), x2)
                nearest_y = min(max(cy, y1), y2)

                distance = math.hypot(
                    cx - nearest_x,
                    cy - nearest_y,
                )

                max_distance = max(
                    100.0,
                    2.5 * max(x2 - x1, y2 - y1),
                )

                if distance <= max_distance:
                    candidates.append((distance, index))

            candidates.sort(key=lambda candidate: candidate[0])

            if not candidates:
                unassociated.append({
                    **parsed,
                    "text": text,
                    "confidence": confidence,
                    "center": (
                        min(max(cx, 0), width),
                        min(max(cy, 0), height),
                    ),
                    "aircraft_index": None,
                })
                continue

            # Reject ambiguous associations between nearby aircraft.
            if (
                len(candidates) > 1
                and abs(candidates[1][0] - candidates[0][0]) < 20
            ):
                print(
                    f"[OCR] Ambiguous aircraft association rejected: {text}",
                    flush=True,
                )
                continue

            aircraft_index = candidates[0][1]
            record = associated_by_aircraft[aircraft_index]

            # Merge separate OCR readings into the same aircraft record.
            for field in ("callsign", "altitude", "heading"):
                value = parsed.get(field)

                if value is None:
                    continue

                # Keep the higher-confidence reading for duplicate fields.
                if (
                    record[field] is None
                    or confidence > record.get(
                        f"{field}_confidence", -1.0
                    )
                ):
                    record[field] = value
                    record[f"{field}_confidence"] = confidence

            record["confidence"] = max(
                record["confidence"],
                confidence,
            )

            record["text"].append(text)

            record["center"] = (
                min(max(cx, 0), width),
                min(max(cy, 0), height),
            )

        # Remove aircraft for which no readable field was associated.
        associated = [
            record
            for record in associated_by_aircraft.values()
            if (
                record["callsign"] is not None
                or record["altitude"] is not None
                or record["heading"] is not None
            )
        ]

        # Preserve unassociated readings for diagnostics, but do not
        # use them to identify an aircraft's highest or lowest altitude.
        associated.extend(unassociated)

        print(
            f"[OCR] Aircraft records with readable fields: "
            f"{len(associated) - len(unassociated)}; "
            f"unassociated text records: {len(unassociated)}",
            flush=True,
        )

        return associated

    def _answer_ocr_question(
        self,
        image_path,
        question,
        detections,
    ):

        lowered = question.lower()

        asks_callsign = any(
            term in lowered
            for term in (
                "callsign",
                "call sign",
                "flight number",
            )
        )

        asks_altitude = (
            "altitude" in lowered
            or "height" in lowered
        )

        asks_heading = (
            "heading" in lowered
            or re.search(r"\bhdg\b", lowered)
        )

        asks_extreme = any(
            term in lowered
            for term in (
                "highest",
                "maximum",
                "max",
                "lowest",
                "minimum",
                "min",
            )
        )

        if not (
            asks_callsign
            or asks_altitude
            or asks_heading
        ):
            return None

        ocr_items = self._ocr_text(
            image_path,
            detections,
        )

        with Image.open(image_path) as image:
            image_size = image.size

        labels = self._associate_ocr(
            detections,
            ocr_items,
            image_size,
        )

        if not labels:

            return {
                "answer": (
                    "I could not reliably read any callsign, "
                    "altitude, or heading labels in this image."
                ),
                "labels": [],
                "ocr_confidence": self._last_ocr_confidence,
                "verdict": "yolo_ocr_unreadable",
            }

        requested_callsign = None

        match = CALLSIGN_PATTERN.search(
            question.upper()
        )

        if match:
            requested_callsign = match.group(0)

        if requested_callsign:

            matching = [
                item
                for item in labels
                if item.get("callsign") == requested_callsign
            ]

            if not matching:

                return {
                    "answer": (
                        f"I could not reliably locate a readable "
                        f"label for callsign {requested_callsign}."
                    ),
                    "labels": labels,
                    "ocr_confidence": self._last_ocr_confidence,
                    "verdict": "yolo_ocr_match_not_found",
                }

            item = max(
                matching,
                key=lambda value: value["confidence"],
            )

            if (
                asks_altitude
                and item.get("altitude") is not None
            ):

                answer = (
                    f"{requested_callsign} is shown at an "
                    f"altitude of {item['altitude']:,} ft."
                )

            elif (
                asks_heading
                and item.get("heading") is not None
            ):

                answer = (
                    f"{requested_callsign} has a heading of "
                    f"{item['heading']}°."
                )

            else:

                answer = (
                    f"I can read callsign {requested_callsign}."
                )

            return {
                "answer": answer,
                "labels": labels,
                "ocr_confidence": item["confidence"],
                "verdict": "yolo_ocr_match",
            }

        if asks_extreme and asks_altitude:

            eligible = [
                item
                for item in labels
                if (
                    item.get("callsign")
                    and item.get("altitude") is not None
                    and item["confidence"] >= 0.50
                )
            ]

            if len(eligible) < 2:

                return {
                    "answer": (
                        "I need at least two reliably read "
                        "aircraft altitude labels to compare "
                        "the highest or lowest altitude."
                    ),
                    "labels": labels,
                    "ocr_confidence": self._last_ocr_confidence,
                    "verdict": "ocr_altitude_insufficient",
                }

            want_highest = (
                "highest" in lowered
                or "maximum" in lowered
                or bool(re.search(r"\bmax\b", lowered))
            )

            chosen = (
                max if want_highest else min
            )(
                eligible,
                key=lambda item: item["altitude"],
            )

            qualifier = (
                "highest" if want_highest else "lowest"
            )

            return {
                "answer": (
                    f"{chosen['callsign']} has the {qualifier} "
                    f"readable altitude at "
                    f"{chosen['altitude']:,} ft."
                ),
                "labels": labels,
                "ocr_confidence": chosen["confidence"],
                "verdict": "ocr_altitude_extreme",
            }

        if asks_callsign and not asks_altitude and not asks_heading:

            callsigns = sorted({
                item["callsign"]
                for item in labels
                if item.get("callsign")
            })

            if callsigns:
                answer = (
                    "Readable callsigns: "
                    + ", ".join(callsigns)
                    + "."
                )
            else:
                answer = (
                    "I could not reliably read a callsign "
                    "in this image."
                )

            return {
                "answer": answer,
                "labels": labels,
                "ocr_confidence": self._last_ocr_confidence,
                "verdict": "yolo_ocr_callsigns",
            }

        readable = []

        for item in labels:

            parts = []

            if item.get("callsign"):
                parts.append(item["callsign"])

            if (
                asks_altitude
                and item.get("altitude") is not None
            ):
                parts.append(
                    f"{item['altitude']:,} ft"
                )

            if (
                asks_heading
                and item.get("heading") is not None
            ):
                parts.append(
                    f"heading {item['heading']}°"
                )

            if parts:
                readable.append(
                    " — ".join(parts)
                )

        if readable:
            answer = (
                "I can read: "
                + "; ".join(readable)
                + "."
            )
        else:
            answer = (
                "I could not reliably read the requested "
                "field from the visible labels."
            )

        return {
            "answer": answer,
            "labels": labels,
            "ocr_confidence": self._last_ocr_confidence,
            "verdict": "yolo_ocr_fields",
        }
    # ------------------------------------------------------------
    # Structured routes for the dataset's question templates (added 4 Nov 2026)
    # ------------------------------------------------------------

    def _answer_structured(self, image_path, question, detections):
        """Answers three question forms from YOLO boxes + OCR instead of the CNN fallback or the
        generic OCR listing (radar_eval 4 Nov: "Is any aircraft heading X?" got an OCR listing,
        "Which quadrant contains aircraft X?" fell to the CNN, which declined 7/10):
          * "Is any aircraft heading <compass>?"            -> OCR headings of all labels
          * "Is there any aircraft in the <compass> quadrant?" -> YOLO boxes around the scope centre
          * "Which quadrant contains aircraft <callsign>?"  -> OCR label -> its YOLO box -> quadrant
        Returns None for any other question."""
        with Image.open(image_path) as image:
            width, height = image.size
        lowered = question.lower()

        m = HEADING_YESNO.search(question)
        if m and not detections:
            return {"answer": "No — I detect no aircraft in this image.", "labels": [],
                    "ocr_confidence": None, "verdict": "yolo_no_aircraft"}
        if m:
            raw = m.group("dir")
            target = COMPASS_WORDS.get(raw.lower().replace(" ", ""), raw.upper())
            labels = self._associate_ocr(detections, self._ocr_text(image_path, detections),
                                         (width, height))
            headings = [item["heading"] for item in labels if item.get("heading") is not None]
            if not headings:
                return {"answer": "I could not read any heading labels in this image.",
                        "labels": labels, "ocr_confidence": self._last_ocr_confidence,
                        "verdict": "ocr_heading_unreadable"}
            matches = [item for item in labels if item.get("heading") is not None
                       and heading_to_compass8(item["heading"]) == target]
            if matches:
                names = ", ".join(f"{i.get('callsign') or 'unlabelled'} ({i['heading']:03d}°)" for i in matches)
                answer = f"Yes — heading {target}: {names}."
            else:
                answer = (f"No — none of the {len(headings)} readable headings is {target}.")
            return {"answer": answer, "labels": labels, "ocr_confidence": self._last_ocr_confidence,
                    "verdict": "ocr_heading_yesno"}

        q = COMPASS_QUADRANT.search(question)
        if q and "quadrant" in lowered and any(w in lowered for w in ("is there", "are there", "any")):
            target = f"{q.group(1).lower()}-{q.group(2).lower()}"
            n = sum(self._compass_quadrant(self._quadrant_for_box(d["bbox"], width, height)) == target
                    for d in detections)
            answer = (f"Yes — I detect {n} aircraft in the {target} quadrant." if n else
                      f"No — I detect no aircraft in the {target} quadrant.")
            return {"answer": answer, "labels": [], "ocr_confidence": None,
                    "verdict": "yolo_compass_quadrant"}

        c = CALLSIGN_PATTERN.search(question.upper())
        if c and "quadrant" in lowered:
            callsign = c.group(0)
            labels = self._associate_ocr(detections, self._ocr_text(image_path, detections),
                                         (width, height))
            hit = [item for item in labels if item.get("callsign") == callsign
                   and item.get("aircraft_index") is not None]
            if not hit:
                return {"answer": f"I could not find a readable label for {callsign} next to a detected aircraft.",
                        "labels": labels, "ocr_confidence": self._last_ocr_confidence,
                        "verdict": "yolo_ocr_match_not_found"}
            item = max(hit, key=lambda i: i["confidence"])
            screen = self._quadrant_for_box(detections[item["aircraft_index"]]["bbox"], width, height)
            return {"answer": f"{callsign} is in the {self._compass_quadrant(screen)} quadrant "
                              f"({screen.replace('-', ' ')} of the scope).",
                    "labels": labels, "ocr_confidence": item["confidence"],
                    "verdict": "ocr_yolo_quadrant"}
        return None

    # ------------------------------------------------------------
    # VQA classification
    # ------------------------------------------------------------

    def _answer_vqa(self, image_path, question):

        image_tensor = self._prepare_image_tensor(image_path)

        question_tensor, question_lengths = (
            self._prepare_question_tensor(question)
        )

        with torch.inference_mode():

            output = self.vqa_model(
                image_tensor,
                question_tensor,
                question_lengths=question_lengths,
            )

            # Use all 28 classifier outputs first.
            logits = output.get("classification_logits")

            if logits is None:
                raise RuntimeError(
                    "RadarVQAModel did not return classification_logits."
                )

            # Validate the output dimension and exclude reserved class 27.
            logits = self.get_valid_logits(logits)

            probabilities = F.softmax(logits, dim=-1)

            confidence, class_index = probabilities.max(dim=-1)

        index = int(class_index.item())
        score = float(confidence.item())

        # Decode using the shared, validated mapping.
        answer = self.decode_answer(index)

        if score < self.decline_threshold:

            answer_text = (
                "I cannot provide a reliable answer to "
                "that question from this image."
            )

            verdict = "vqa_low_confidence"

        elif score < self.hedge_threshold:

            answer_text = (
                f"My best estimate is: {answer}. "
                "The model's confidence is low, so please "
                "treat this as uncertain."
            )

            verdict = "vqa_hedged"

        else:

            answer_text = str(answer)
            verdict = "vqa_answer"

        return {
            "answer": answer_text,
            "confidence": score,
            "verdict": verdict,
        }

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------

    def _log_event(self, payload):

        path = os.path.join(
            self.logs_dir,
            "radar_inference.jsonl",
        )

        record = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            **payload,
        }

        try:

            with open(
                path,
                "a",
                encoding="utf-8",
            ) as file:

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        default=str,
                    ) + "\n"
                )

        except Exception:

            self.logger.exception(
                "Could not write radar inference log"
            )

    # ------------------------------------------------------------
    # Public API used by integrated.py
    # ------------------------------------------------------------

    def handle_question(
        self,
        image_path: str,
        question: str,
    ) -> Dict[str, Any]:

        """Analyze one uploaded radar image and answer a question."""

        if (
            not image_path
            or not os.path.isfile(image_path)
        ):

            return {
                "kind": "vqa",
                "answer": (
                    "Please provide a valid radar image "
                    "before asking a question."
                ),
                "image": None,
                "confidence": None,
                "confidence_type": None,
                "ocr_confidence": None,
                "verdict": "image_missing",
            }

        if not question or not question.strip():

            return {
                "kind": "vqa",
                "answer": (
                    "Please enter a question about "
                    "the radar image."
                ),
                "image": image_path,
                "confidence": None,
                "confidence_type": None,
                "ocr_confidence": None,
                "verdict": "question_missing",
            }

        if not self._is_radar_question(question):

            return {
                "kind": "vqa",
                "answer": (
                    "That question does not appear to be "
                    "about the aircraft radar image. Please "
                    "ask about detected aircraft, their "
                    "positions, callsigns, altitudes, or headings."
                ),
                "image": image_path,
                "confidence": None,
                "confidence_type": None,
                "ocr_confidence": None,
                "verdict": "out_of_scope",
            }

        try:

            # Step 1: Detect aircraft using YOLOv5.
            detections = self._detect_aircraft(
                image_path
            )

            detector_confidence = (
                self._mean_detector_confidence(
                    detections
                )
            )
            # Step 1A (4 Nov 2026): heading yes/no, compass-quadrant presence, quadrant of a callsign.
            structured = self._answer_structured(image_path, question, detections)
            if structured is not None:
                annotated = self._annotate(image_path, detections, structured.get("labels"))
                response = {
                    "kind": "vqa",
                    "answer": structured["answer"],
                    "image": annotated,
                    "confidence": detector_confidence,
                    "confidence_type": "detector",
                    "ocr_confidence": structured.get("ocr_confidence"),
                    "verdict": structured["verdict"],
                    "detections": detections,
                }
                self._log_event({"question": question, "verdict": response["verdict"],
                                 "detection_count": len(detections), "confidence": detector_confidence,
                                 "ocr_confidence": response["ocr_confidence"]})
                return response

            # Step 1B: Route selected questions directly to trained VQA.
            vqa_keywords = (
                "does the image contain aircraft",
                "are multiple aircraft visible",
                "is the radar image clear or cluttered",
            )

            normalized_question = question.lower().strip()

            if any(q in normalized_question for q in vqa_keywords):

                vqa_result = self._answer_vqa(
                    image_path,
                    question,
                )

                annotated = self._annotate(
                    image_path,
                    detections,
                )

                return {
                    "kind": "vqa",
                    "answer": vqa_result["answer"],
                    "image": annotated,
                    "confidence": vqa_result["confidence"],
                    "confidence_type": "vqa",
                    "ocr_confidence": None,
                    "verdict": vqa_result["verdict"],
                    "detections": detections,
                }
            # Step 2: Answer count and position questions
            # directly from YOLO bounding boxes.
            yolo_answer = self._answer_yolo(
                image_path,
                question,
                detections,
            )

            if yolo_answer is not None:

                annotated = self._annotate(
                    image_path,
                    detections,
                )

                response = {
                    "kind": "vqa",
                    "answer": yolo_answer,
                    "image": annotated,
                    "confidence": detector_confidence,
                    "confidence_type": "detector",
                    "ocr_confidence": None,
                    "verdict": "yolo_geometry",
                    "detections": detections,
                }

                self._log_event({
                    "question": question,
                    "verdict": response["verdict"],
                    "detection_count": len(detections),
                    "confidence": detector_confidence,
                })

                return response

            # Step 3: Use PaddleOCR for callsign,
            # altitude and heading questions.
            ocr_result = self._answer_ocr_question(
                image_path,
                question,
                detections,
            )

            if ocr_result is not None:

                annotated = self._annotate(
                    image_path,
                    detections,
                    ocr_result.get("labels"),
                )

                lower = question.lower()

                locate_words = (
                    "locate",
                    "find",
                    "where",
                    "highlight",
                    "show",
                )

                kind = (
                    "locate"
                    if any(
                        word in lower
                        for word in locate_words
                    )
                    else "vqa"
                )

                response = {
                    "kind": kind,
                    "answer": ocr_result["answer"],
                    "image": annotated,
                    "confidence": detector_confidence,
                    "confidence_type": "detector",
                    "ocr_confidence": ocr_result.get(
                        "ocr_confidence"
                    ),
                    "verdict": ocr_result.get(
                        "verdict"
                    ),
                    "detections": detections,
                    "ocr_labels": ocr_result.get(
                        "labels", []
                    ),
                }

                self._log_event({
                    "question": question,
                    "verdict": response["verdict"],
                    "detection_count": len(detections),
                    "confidence": detector_confidence,
                    "ocr_confidence": response[
                        "ocr_confidence"
                    ],
                })

                return response

            # Step 4: Use the trained RadarVQAModel for
            # other image-question classification tasks.
            vqa_result = self._answer_vqa(
                image_path,
                question,
            )

            annotated = self._annotate(
                image_path,
                detections,
            )

            response = {
                "kind": "vqa",
                "answer": vqa_result["answer"],
                "image": annotated,
                "confidence": vqa_result["confidence"],
                "confidence_type": "vqa",
                "ocr_confidence": None,
                "verdict": vqa_result["verdict"],
                "detections": detections,
            }

            self._log_event({
                "question": question,
                "verdict": response["verdict"],
                "detection_count": len(detections),
                "confidence": response["confidence"],
            })

            return response

        except Exception as exc:

            self.logger.exception(
                "Radar analysis failed"
            )

            return {
                "kind": "vqa",
                "answer": (
                    "Radar analysis could not be completed. "
                    f"Technical detail: {type(exc).__name__}: {exc}"
                ),
                "image": image_path,
                "confidence": None,
                "confidence_type": None,
                "ocr_confidence": None,
                "verdict": "analysis_error",
            }