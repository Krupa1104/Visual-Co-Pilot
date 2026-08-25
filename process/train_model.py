r"""
train_model.py
Trains a multi-head CNN on the consolidated Visual Co-Pilot dataset.

Input:  full cockpit panel image
Output: 5 flight parameters
  - altitude   (regression)
  - airspeed   (regression)
  - heading    (regression, via sin/cos to handle 0/360 wraparound)
  - pitch_bin  (classification)
  - roll_bin   (classification)

Usage:
  python train_model.py --data_dir "C:\Users\krupa\sem6\capstone\datasets\asritha\consolidated" --epochs 25
"""

import os
import json
import argparse
import random
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

# =========================================================================
# REPRODUCIBILITY — fix every source of randomness so back-to-back runs
# (e.g. comparing epoch counts, augmentation, loss weights) are an actual
# apples-to-apples comparison instead of different random dice rolls.
# =========================================================================
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Forces deterministic (if sometimes slightly slower) CUDA kernels
    # instead of cuDNN's auto-tuned, run-to-run-varying algorithms.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# =========================================================================
# GLOBAL CONSTANTS & LABEL MAPS
# =========================================================================
PITCH_BIN_EDGES = [-3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
ROLL_BIN_EDGES = [-10, -6, -2, 2, 6, 10]


def bin_label(value, edges):
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= value < hi or (i == len(edges) - 2 and value == hi):
            return f"{lo}-{hi}"
    return "out_of_range"


PITCH_LABELS = [f"{lo}-{hi}" for lo, hi in zip(PITCH_BIN_EDGES[:-1], PITCH_BIN_EDGES[1:])]
ROLL_LABELS = [f"{lo}-{hi}" for lo, hi in zip(ROLL_BIN_EDGES[:-1], ROLL_BIN_EDGES[1:])]
PITCH_LABEL_TO_IDX = {label: i for i, label in enumerate(PITCH_LABELS)}
ROLL_LABEL_TO_IDX = {label: i for i, label in enumerate(ROLL_LABELS)}

NUM_PITCH_CLASSES = len(PITCH_LABELS)
NUM_ROLL_CLASSES = len(ROLL_LABELS)


# =========================================================================
# DATASET
# =========================================================================
class CockpitDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform, norm_stats):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform
        self.norm_stats = norm_stats

        self.df["pitch_bin"] = self.df["pitch"].apply(lambda v: bin_label(v, PITCH_BIN_EDGES))
        self.df["roll_bin"] = self.df["roll"].apply(lambda v: bin_label(v, ROLL_BIN_EDGES))

        bad = self.df[(self.df["pitch_bin"] == "out_of_range") | (self.df["roll_bin"] == "out_of_range")]
        if len(bad) > 0:
            print(f"WARNING: {len(bad)} rows have out-of-range pitch/roll bins in {csv_path}.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        altitude_norm = (row["altitude"] - self.norm_stats["altitude"]["mean"]) / self.norm_stats["altitude"]["std"]
        airspeed_norm = (row["airspeed"] - self.norm_stats["airspeed"]["mean"]) / self.norm_stats["airspeed"]["std"]

        heading_rad = np.deg2rad(row["heading"])
        heading_sin = np.sin(heading_rad)
        heading_cos = np.cos(heading_rad)

        pitch_idx = PITCH_LABEL_TO_IDX[row["pitch_bin"]]
        roll_idx = ROLL_LABEL_TO_IDX[row["roll_bin"]]

        targets = {
            "altitude": torch.tensor(altitude_norm, dtype=torch.float32),
            "airspeed": torch.tensor(airspeed_norm, dtype=torch.float32),
            "heading_sincos": torch.tensor([heading_sin, heading_cos], dtype=torch.float32),
            "pitch_idx": torch.tensor(pitch_idx, dtype=torch.long),
            "roll_idx": torch.tensor(roll_idx, dtype=torch.long),
        }
        return img, targets


# =========================================================================
# MODEL ARCHITECTURE
# =========================================================================
class CockpitNet(nn.Module):
    def __init__(self, num_pitch_classes, num_roll_classes):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
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


# =========================================================================
# MAIN EXECUTION GUARD
# =========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder containing images/, train.csv, val.csv, test.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_width", type=int, default=224)
    parser.add_argument("--img_height", type=int, default=224)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--reg_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducible runs")
    parser.add_argument("--patience", type=int, default=5,
                         help="Stop automatically if val loss doesn't improve for this many epochs in a row")
    args = parser.parse_args()

    set_seed(args.seed)  # re-seed in case --seed overrides the module-level default

    OUTPUT_DIR = args.output_dir or os.path.join(args.data_dir, "model_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    IMAGES_DIR = os.path.join(args.data_dir, "images")
    TRAIN_CSV = os.path.join(args.data_dir, "train.csv")
    VAL_CSV = os.path.join(args.data_dir, "val.csv")
    TEST_CSV = os.path.join(args.data_dir, "test.csv")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Seed: {args.seed}")

    print(f"Pitch classes ({NUM_PITCH_CLASSES}): {PITCH_LABELS}")
    print(f"Roll classes ({NUM_ROLL_CLASSES}): {ROLL_LABELS}")

    # Normalization Stats
    train_df = pd.read_csv(TRAIN_CSV)
    norm_stats = {
        "altitude": {"mean": float(train_df["altitude"].mean()), "std": float(train_df["altitude"].std()) or 1.0},
        "airspeed": {"mean": float(train_df["airspeed"].mean()), "std": float(train_df["airspeed"].std()) or 1.0},
    }
    print("Normalization stats (from train split):", norm_stats)

    with open(os.path.join(OUTPUT_DIR, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "label_maps.json"), "w") as f:
        json.dump({
            "pitch_labels": PITCH_LABELS,
            "roll_labels": ROLL_LABELS,
            "pitch_bin_edges": PITCH_BIN_EDGES,
            "roll_bin_edges": ROLL_BIN_EDGES,
        }, f, indent=2)

    # Transforms & DataLoaders
    img_transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CockpitDataset(TRAIN_CSV, IMAGES_DIR, img_transform, norm_stats)
    val_ds = CockpitDataset(VAL_CSV, IMAGES_DIR, img_transform, norm_stats)
    test_ds = CockpitDataset(TEST_CSV, IMAGES_DIR, img_transform, norm_stats)

    # Seeded generator so the shuffle order itself is reproducible too,
    # not just the model init.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # Setup Training
    model = CockpitNet(NUM_PITCH_CLASSES, NUM_ROLL_CLASSES).to(DEVICE)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    def compute_loss(preds, targets):
        loss_altitude = mse_loss(preds["altitude"], targets["altitude"])
        loss_airspeed = mse_loss(preds["airspeed"], targets["airspeed"])
        loss_heading = mse_loss(preds["heading_sincos"], targets["heading_sincos"])
        loss_pitch = ce_loss(preds["pitch_logits"], targets["pitch_idx"])
        loss_roll = ce_loss(preds["roll_logits"], targets["roll_idx"])

        reg_loss = loss_altitude + loss_airspeed + loss_heading
        cls_loss = loss_pitch + loss_roll
        total = args.reg_weight * reg_loss + args.cls_weight * cls_loss

        return total, {
            "altitude": loss_altitude.item(),
            "airspeed": loss_airspeed.item(),
            "heading": loss_heading.item(),
            "pitch": loss_pitch.item(),
            "roll": loss_roll.item(),
            "total": total.item(),
        }

    def run_epoch(loader, train_mode):
        model.train() if train_mode else model.eval()
        totals = Counter()
        pitch_correct, roll_correct, n = 0, 0, 0

        context = torch.enable_grad() if train_mode else torch.no_grad()
        with context:
            for imgs, targets in loader:
                imgs = imgs.to(DEVICE)
                targets = {k: v.to(DEVICE) for k, v in targets.items()}

                preds = model(imgs)
                loss, loss_parts = compute_loss(preds, targets)

                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                for k, v in loss_parts.items():
                    totals[k] += v * imgs.size(0)

                pitch_correct += (preds["pitch_logits"].argmax(1) == targets["pitch_idx"]).sum().item()
                roll_correct += (preds["roll_logits"].argmax(1) == targets["roll_idx"]).sum().item()
                n += imgs.size(0)

        avg_losses = {k: v / n for k, v in totals.items()}
        avg_losses["pitch_acc"] = pitch_correct / n
        avg_losses["roll_acc"] = roll_correct / n
        return avg_losses

    # Train Loop — runs up to --epochs, but stops early once val loss hasn't
    # improved for --patience epochs in a row, so you don't have to guess
    # the right epoch count by hand (and re-run to find out you guessed wrong).
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_loader, train_mode=True)
        val_metrics = run_epoch(val_loader, train_mode=False)
        scheduler.step(val_metrics["total"])

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  train: total={train_metrics['total']:.4f} "
              f"alt={train_metrics['altitude']:.4f} spd={train_metrics['airspeed']:.4f} "
              f"hdg={train_metrics['heading']:.4f} pitch_acc={train_metrics['pitch_acc']:.3f} "
              f"roll_acc={train_metrics['roll_acc']:.3f}")
        print(f"  val:   total={val_metrics['total']:.4f} "
              f"alt={val_metrics['altitude']:.4f} spd={val_metrics['airspeed']:.4f} "
              f"hdg={val_metrics['heading']:.4f} pitch_acc={val_metrics['pitch_acc']:.3f} "
              f"roll_acc={val_metrics['roll_acc']:.3f}")

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
            }, best_ckpt_path)
            print(f"  -> saved new best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  (no improvement for {epochs_without_improvement}/{args.patience} epochs)")
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} — val loss hasn't improved in {args.patience} epochs.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")

    # Final Test Evaluation
    print("\nEvaluating best checkpoint on test set...")
    checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(test_loader, train_mode=False)

    alt_std = norm_stats["altitude"]["std"]
    spd_std = norm_stats["airspeed"]["std"]
    print(f"\nTest results:")
    print(f"  Altitude MSE (normalized): {test_metrics['altitude']:.4f}  "
          f"(~{(test_metrics['altitude'] ** 0.5) * alt_std:.1f} ft typical error)")
    print(f"  Airspeed MSE (normalized): {test_metrics['airspeed']:.4f}  "
          f"(~{(test_metrics['airspeed'] ** 0.5) * spd_std:.1f} kt typical error)")
    print(f"  Heading sin/cos MSE: {test_metrics['heading']:.4f}")
    print(f"  Pitch bin accuracy: {test_metrics['pitch_acc']:.3f}")
    print(f"  Roll bin accuracy: {test_metrics['roll_acc']:.3f}")

    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("  - best_model.pt       (model weights)")
    print("  - norm_stats.json     (needed to un-normalize predictions at inference)")
    print("  - label_maps.json     (pitch/roll bin label <-> class index mapping)")
    print("  - test_results.json   (final test set metrics)")