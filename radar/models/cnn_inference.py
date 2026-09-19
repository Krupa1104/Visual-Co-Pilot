"""
CNN Inference Engine
Pure pixel reading — no CLIP, no ChromaDB, no embeddings.
ResNet50 reads image pixels directly and answers questions.
"""

import json, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from typing import Dict, List


class CNNInferenceEngine:
    CHARS    = list("abcdefghijklmnopqrstuvwxyz0123456789 ?,'.")
    CHAR2IDX = {c: i+1 for i, c in enumerate(CHARS)}
    MAX_Q    = 60

    def __init__(self, model_path: str,
                 answer_index_path: str,
                 device: str = "cpu"):
        self.device = device

        with open(answer_index_path) as f:
            self.answer_index = json.load(f)

        # Import here to avoid circular import
        from models.cnn_model import CNNVQAModel
        self.model = CNNVQAModel(
            num_classes=len(self.answer_index),
            device=device)
        self.model.load(model_path)
        self.model.eval()

        # Pixel transform — ImageNet normalisation
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225])])

    def answer(self, image: Image.Image,
               question: str) -> Dict:
        """
        Step 1: Read raw pixels from image (224x224x3)
        Step 2: Pass through ResNet50 CNN
        Step 3: Combine with question via cross-attention
        Step 4: Return answer + confidence
        """
        # Read pixels
        img_t = self.tf(image.convert("RGB")).unsqueeze(0)

        # Tokenise question as characters
        q     = question.lower()[:self.MAX_Q]
        q_tok = [self.CHAR2IDX.get(c, 0) for c in q]
        q_tok += [0] * (self.MAX_Q - len(q_tok))
        q_t   = torch.tensor([q_tok], dtype=torch.long)

        # CNN reads pixels → answer
        answers, confs = self.model.predict(
            img_t, q_t, self.answer_index)

        return {
            "question":   question,
            "answer":     answers[0],
            "confidence": round(confs[0], 4),
            "route":      "cnn_pixel_reading",
            "source":     "ResNet50",
        }

    def batch_answer(self, image: Image.Image,
                     questions: List[str]) -> List[Dict]:
        return [self.answer(image, q) for q in questions]
