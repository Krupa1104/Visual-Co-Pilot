
"""
Radar VQA PyTorch Dataset.

Loads:
- Radar images from data/synthetic_complete/images
- QA records from data/splits/{train,val,test}.jsonl
- Closed-set labels from configs/answer_mapping.py

Callsign answers remain text sequences for the separate OCR branch.
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.answer_mapping import encode_answer


IMAGE_DIR = ROOT / "data" / "synthetic_complete" / "images"

IMAGE_SIZE = 1280

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
def encode_ocr_sequence(callsign: str) -> tuple[list[int], list[int]]:
    """
    Prepare teacher-forcing input and target sequences.

    Example:
        QTR811
        Input:  BOS Q T R 8 1 1
        Target: Q   T R 8 1 1 EOS
    """
    from src.models.radar_vqa_model import (
        BOS_ID,
        EOS_ID,
        encode_text,
    )

    character_ids = encode_text(callsign)

    if not character_ids:
        raise ValueError("Callsign cannot be empty.")

    input_ids = [BOS_ID] + character_ids
    target_ids = character_ids + [EOS_ID]

    return input_ids, target_ids

class RadarVQADataset(Dataset):
    def __init__(self, split, transform=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        self.split = split
        self.transform = transform or IMAGE_TRANSFORM

        self.qa_path = ROOT / "data" / "splits" / f"{split}.jsonl"

        if not self.qa_path.is_file():
            raise FileNotFoundError(self.qa_path)

        self.records = []

        with self.qa_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                record = json.loads(line)

                required = {
                    "qa_id",
                    "image_id",
                    "image_path",
                    "question_type",
                    "question",
                    "answer",
                }

                missing = required - record.keys()
                if missing:
                    raise ValueError(
                        f"Line {line_number}: missing fields {missing}"
                    )

                self.records.append(record)

        if not self.records:
            raise ValueError(f"No QA records found in {self.qa_path}")

        self._validate_records()

    def _validate_records(self):
        seen_ids = set()

        for record in self.records:
            if record["qa_id"] in seen_ids:
                raise ValueError(
                    f"Duplicate QA ID: {record['qa_id']}"
                )

            seen_ids.add(record["qa_id"])

            if record["question_type"] == "callsign":
                if not record["answer"].strip():
                    raise ValueError(
                        f"Empty callsign answer: {record['qa_id']}"
                    )
            else:
                # Confirms the answer exists in the shared mapping.
                encode_answer(
                    record["question_type"],
                    record["answer"],
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image_path = ROOT / record["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        question_type = record["question_type"]
        answer = record["answer"]

        is_ocr = question_type == "callsign"

        if is_ocr:
            class_id = -1
            ocr_target = answer
        else:
            class_id = encode_answer(question_type, answer)
            ocr_target = ""

        return {
            "image": image,
            "question": record["question"],
            "question_type": question_type,
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "ocr_target": ocr_target,
            "is_ocr": torch.tensor(is_ocr, dtype=torch.bool),
            "image_id": record["image_id"],
            "qa_id": record["qa_id"],
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    for split in ("train", "val", "test"):
        dataset = RadarVQADataset(split)

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(loader))

        print(f"\n{split.upper()} DATASET")
        print(f"Records: {len(dataset):,}")
        print(f"Image batch shape: {batch['image'].shape}")
        print(f"Class IDs: {batch['class_id'].tolist()}")
        print(f"Question types: {batch['question_type']}")
        print(f"OCR targets: {batch['ocr_target']}")
