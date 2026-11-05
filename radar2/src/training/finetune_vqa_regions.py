
from __future__ import annotations

import json
import random
import sys
from itertools import zip_longest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.radar_vqa_model import RadarVQAModel, encode_text, PAD_ID
from src.dataset.radar_dataset import RadarVQADataset
from src.dataset.ocr_training_dataset import CallsignOCRDataset


# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_ROOT = Path("/content/drive/MyDrive/Final_Radar_Training")

INITIAL_CHECKPOINT = DRIVE_ROOT / "vqa_runs" / "best.pt"
REGION_CACHE_DIR = DRIVE_ROOT / "region_cache"

OUTPUT_DIR = DRIVE_ROOT / "vqa_region_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_REGIONS = 8
IMAGE_SIZE = 384
CROP_SIZE = 224

BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
NUM_WORKERS = 2

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# REGION-AWARE COLLATION
# ============================================================

def collate_vqa(batch):
    images = torch.stack([item["image"] for item in batch])

    questions = [
        encode_text(item["question"])
        for item in batch
    ]

    max_question_length = max(len(q) for q in questions)

    question_ids = torch.full(
        (len(batch), max_question_length),
        PAD_ID,
        dtype=torch.long,
    )

    question_lengths = torch.tensor(
        [len(q) for q in questions],
        dtype=torch.long,
    )

    for index, ids in enumerate(questions):
        question_ids[index, :len(ids)] = torch.tensor(ids)

    labels = torch.stack([
        item["class_id"] for item in batch
    ])

    # Load cached YOLO aircraft crops using image_id.
    cached_regions = []

    for item in batch:
        cache_path = REGION_CACHE_DIR / f"{item['image_id']}.pt"

        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing region cache: {cache_path}"
            )

        crops = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        if crops.ndim != 4 or crops.shape[1:] != (
            3, CROP_SIZE, CROP_SIZE
        ):
            raise ValueError(
                f"Invalid crop tensor in {cache_path}: "
                f"{tuple(crops.shape)}"
            )

        # YOLO detections are stored in detector result order.
        crops = crops[:MAX_REGIONS]
        cached_regions.append(crops)

    # At least one padded slot is required by the model.
    max_regions = max(
        1,
        max(crops.shape[0] for crops in cached_regions),
    )

    aircraft_crops = torch.zeros(
        len(batch),
        max_regions,
        3,
        CROP_SIZE,
        CROP_SIZE,
        dtype=torch.float32,
    )

    # True means that a crop position is padding.
    aircraft_mask = torch.ones(
        len(batch),
        max_regions,
        dtype=torch.bool,
    )

    for index, crops in enumerate(cached_regions):
        count = crops.shape[0]

        if count > 0:
            aircraft_crops[index, :count] = crops
            aircraft_mask[index, :count] = False

    return {
        "images": images,
        "question_ids": question_ids,
        "question_lengths": question_lengths,
        "labels": labels,
        "aircraft_crops": aircraft_crops,
        "aircraft_mask": aircraft_mask,
    }


# ============================================================
# DATA LOADERS
# ============================================================

def make_loaders():
    loaders = {}

    for split in ("train", "val"):
        dataset = RadarVQADataset(
            split,
            transform=IMAGE_TRANSFORM,
        )

        # Callsign questions are handled by the OCR branch.
        dataset.records = [
            record
            for record in dataset.records
            if record["question_type"] != "callsign"
        ]

        loaders[split] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_vqa,
            persistent_workers=(NUM_WORKERS > 0),
        )

    ocr_loaders = {}

    for split in ("train", "val"):
        dataset = CallsignOCRDataset(split)

        ocr_loaders[split] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0),
        )

    return loaders, ocr_loaders


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def load_model():
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Initial VQA checkpoint not found: {INITIAL_CHECKPOINT}"
        )

    # pretrained=False avoids downloading ImageNet weights.
    # The existing checkpoint supplies all model parameters.
    model = RadarVQAModel(pretrained=False)

    checkpoint = torch.load(
        INITIAL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE)

    print("Loaded existing VQA checkpoint:", INITIAL_CHECKPOINT)

    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print("Original checkpoint epoch:", checkpoint["epoch"])

    return model


# ============================================================
# CLASSIFICATION TRAINING
# ============================================================

def classification_step(model, batch, optimizer, scaler):
    images = batch["images"].to(DEVICE, non_blocking=True)
    questions = batch["question_ids"].to(
        DEVICE, non_blocking=True
    )
    lengths = batch["question_lengths"]
    labels = batch["labels"].to(DEVICE, non_blocking=True)

    crops = batch["aircraft_crops"].to(
        DEVICE, non_blocking=True
    )
    crop_mask = batch["aircraft_mask"].to(
        DEVICE, non_blocking=True
    )

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(DEVICE.type == "cuda"),
    ):
        outputs = model(
            images,
            questions,
            lengths,
            aircraft_crops=crops,
            aircraft_mask=crop_mask,
        )

        loss = nn.functional.cross_entropy(
            outputs["valid_class_logits"],
            labels,
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    scaler.step(optimizer)
    scaler.update()

    predictions = outputs["valid_class_logits"].argmax(dim=1)
    correct = (predictions == labels).sum().item()

    return loss.item(), correct, labels.size(0)


# ============================================================
# OCR TRAINING
# ============================================================

def ocr_step(model, batch, optimizer, scaler):
    images = batch["ocr_image"].to(
        DEVICE, non_blocking=True
    )

    input_ids = batch["ocr_input_ids"].to(
        DEVICE, non_blocking=True
    )

    targets = batch["ocr_target_ids"].to(
        DEVICE, non_blocking=True
    )

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(DEVICE.type == "cuda"),
    ):
        visual_features = model.visual_encoder(images)

        logits = model.ocr_decoder(
            visual_features,
            input_ids,
        )

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=PAD_ID,
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    scaler.step(optimizer)
    scaler.update()

    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()

    return loss.item(), correct, total


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, vqa_loader, ocr_loader):
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for batch in tqdm(vqa_loader, desc="VQA validation"):
        images = batch["images"].to(DEVICE)
        questions = batch["question_ids"].to(DEVICE)
        lengths = batch["question_lengths"]
        labels = batch["labels"].to(DEVICE)

        crops = batch["aircraft_crops"].to(DEVICE)
        crop_mask = batch["aircraft_mask"].to(DEVICE)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda"),
        ):
            outputs = model(
                images,
                questions,
                lengths,
                aircraft_crops=crops,
                aircraft_mask=crop_mask,
            )

            loss = nn.functional.cross_entropy(
                outputs["valid_class_logits"],
                labels,
            )

        predictions = outputs["valid_class_logits"].argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    vqa_accuracy = total_correct / max(total_samples, 1)
    vqa_loss = total_loss / max(total_samples, 1)

    # Evaluate OCR token accuracy separately.
    ocr_correct = 0
    ocr_total = 0

    for batch in tqdm(ocr_loader, desc="OCR validation"):
        images = batch["ocr_image"].to(DEVICE)
        input_ids = batch["ocr_input_ids"].to(DEVICE)
        targets = batch["ocr_target_ids"].to(DEVICE)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda"),
        ):
            features = model.visual_encoder(images)
            logits = model.ocr_decoder(features, input_ids)

        predictions = logits.argmax(dim=-1)

        valid = targets.ne(PAD_ID)

        ocr_correct += (
            (predictions == targets) & valid
        ).sum().item()

        ocr_total += valid.sum().item()

    ocr_accuracy = ocr_correct / max(ocr_total, 1)

    return vqa_loss, vqa_accuracy, ocr_accuracy


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    set_seed(SEED)

    print("=" * 65)
    print("REGION-AWARE RADAR VQA FINE-TUNING")
    print("=" * 65)
    print("Device:", DEVICE)
    print("Maximum aircraft crops:", MAX_REGIONS)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    vqa_loaders, ocr_loaders = make_loaders()

    model = load_model()

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {trainable:,}")
    print("VQA training batches:", len(vqa_loaders["train"]))
    print("OCR training batches:", len(ocr_loaders["train"]))

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    best_accuracy = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        vqa_loss_sum = 0.0
        vqa_correct = 0
        vqa_total = 0

        ocr_loss_sum = 0.0
        ocr_correct = 0
        ocr_total = 0

        vqa_iterator = iter(vqa_loaders["train"])
        ocr_iterator = iter(ocr_loaders["train"])

        total_steps = max(
            len(vqa_loaders["train"]),
            len(ocr_loaders["train"]),
        )

        progress = tqdm(
            range(total_steps),
            desc=f"Epoch {epoch}/{EPOCHS}",
        )

        for _ in progress:
            vqa_batch = next(vqa_iterator, None)
            ocr_batch = next(ocr_iterator, None)

            if vqa_batch is not None:
                loss, correct, count = classification_step(
                    model,
                    vqa_batch,
                    optimizer,
                    scaler,
                )

                vqa_loss_sum += loss * count
                vqa_correct += correct
                vqa_total += count

            if ocr_batch is not None:
                loss, correct, count = ocr_step(
                    model,
                    ocr_batch,
                    optimizer,
                    scaler,
                )

                ocr_loss_sum += loss * count
                ocr_correct += correct
                ocr_total += count

            progress.set_postfix({
                "VQA loss": round(
                    vqa_loss_sum / max(vqa_total, 1), 4
                ),
                "VQA acc": round(
                    vqa_correct / max(vqa_total, 1), 4
                ),
                "OCR loss": round(
                    ocr_loss_sum / max(ocr_total, 1), 4
                ),
            })

        val_loss, val_accuracy, val_ocr_accuracy = evaluate(
            model,
            vqa_loaders["val"],
            ocr_loaders["val"],
        )

        epoch_result = {
            "epoch": epoch,
            "train_vqa_loss": vqa_loss_sum / max(vqa_total, 1),
            "train_vqa_accuracy": vqa_correct / max(vqa_total, 1),
            "train_ocr_loss": ocr_loss_sum / max(ocr_total, 1),
            "train_ocr_token_accuracy": (
                ocr_correct / max(ocr_total, 1)
            ),
            "val_vqa_loss": val_loss,
            "val_vqa_accuracy": val_accuracy,
            "val_ocr_token_accuracy": val_ocr_accuracy,
        }

        history.append(epoch_result)

        print("\nEpoch results:")
        print(json.dumps(epoch_result, indent=2))

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_vqa_accuracy": val_accuracy,
            "val_ocr_token_accuracy": val_ocr_accuracy,
            "history": history,
            "max_regions": MAX_REGIONS,
            "architecture": "ResNet50 region-aware Radar VQA",
        }

        torch.save(
            checkpoint,
            OUTPUT_DIR / "last.pt",
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save(
                checkpoint,
                OUTPUT_DIR / "best.pt",
            )

            print("Saved new best checkpoint.")

        with (OUTPUT_DIR / "history.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(history, file, indent=2)

        print("Checkpoint directory:", OUTPUT_DIR)

    print("\nFine-tuning completed.")
    print("Best validation VQA accuracy:", best_accuracy)
    print("Best checkpoint:", OUTPUT_DIR / "best.pt")


if __name__ == "__main__":
    main()
