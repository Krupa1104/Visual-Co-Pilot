
"""
PyTorch dataset for callsign OCR training.

Reads:
- data/ocr_crops/{train,val,test}/manifest.jsonl
- Pre-extracted 224x96 callsign crop images

Does NOT read annotation JSON files.
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

from src.dataset.radar_dataset import encode_ocr_sequence


OCR_DIR = ROOT / "data" / "ocr_crops"

OCR_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class CallsignOCRDataset(Dataset):

    def __init__(self, split, transform=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        self.split = split
        self.transform = transform or OCR_TRANSFORM

        self.manifest_path = (
            OCR_DIR / split / "manifest.jsonl"
        )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)

        self.records = []

        with self.manifest_path.open(
            "r", encoding="utf-8"
        ) as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                record = json.loads(line)

                required = {
                    "image_id",
                    "callsign",
                    "crop_path",
                }

                missing = required - record.keys()

                if missing:
                    raise ValueError(
                        f"Line {line_number}: missing fields {missing}"
                    )

                callsign = record["callsign"]

                if len(callsign) != 6:
                    raise ValueError(
                        f"Invalid callsign at line {line_number}: "
                        f"{callsign}"
                    )

                if not callsign.isalnum() or not callsign.isupper():
                    raise ValueError(
                        f"Invalid callsign characters: {callsign}"
                    )

                crop_path = ROOT / record["crop_path"]

                if not crop_path.is_file():
                    raise FileNotFoundError(crop_path)

                self.records.append({
                    "image_id": record["image_id"],
                    "callsign": callsign,
                    "crop_path": crop_path,
                })

        if not self.records:
            raise ValueError(
                f"No OCR records found for split: {split}"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(record["crop_path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        input_ids, target_ids = encode_ocr_sequence(
            record["callsign"]
        )

        return {
            "ocr_image": image,
            "ocr_input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "ocr_target_ids": torch.tensor(
                target_ids,
                dtype=torch.long,
            ),
            "callsign": record["callsign"],
            "image_id": record["image_id"],
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    for split in ("train", "val", "test"):
        dataset = CallsignOCRDataset(split)

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(loader))

        print(f"\n{split.upper()} OCR DATASET")
        print(f"Records: {len(dataset):,}")
        print(f"Image batch: {batch['ocr_image'].shape}")
        print(f"Input IDs: {batch['ocr_input_ids'].shape}")
        print(f"Target IDs: {batch['ocr_target_ids'].shape}")
        print(f"Callsigns: {batch['callsign']}")
        print(
            "Input/target lengths match:",
            batch["ocr_input_ids"].shape[1]
            == batch["ocr_target_ids"].shape[1],
        )
