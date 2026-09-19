r"""
full_test_evaluation.py
Rigorous evaluation on the FULL test set (not a hand-picked sample):
  1. Phase inference accuracy + confusion matrix (predicted phase vs true phase)
  2. pitch_bin classification confusion matrix
  3. roll_bin classification confusion matrix

Replaces the earlier "5/5 correct" anecdotal check (which was tuned to those
5 examples, not validated against them) with a real, defensible number
computed against all 293 held-out test rows.

Usage:
  python full_test_evaluation.py --data_dir "C:\...\consolidated"
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--img_width", type=int, default=224)
parser.add_argument("--img_height", type=int, default=224)
parser.add_argument("--use_letterbox", action="store_true",
                     help="Only set this if best_model.pt was trained with letterbox "
                          "preprocessing. Default (off) matches the original epoch-16 "
                          "checkpoint, which used plain stretch resize.")
args = parser.parse_args()

MODEL_DIR = os.path.join(args.data_dir, "model_output")
IMAGES_DIR = os.path.join(args.data_dir, "images")
TEST_CSV = os.path.join(args.data_dir, "test.csv")
OUTPUT_DIR = os.path.join(MODEL_DIR, "full_evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(MODEL_DIR, "norm_stats.json")) as f:
    norm_stats = json.load(f)
with open(os.path.join(MODEL_DIR, "label_maps.json")) as f:
    label_maps = json.load(f)

PITCH_LABELS = label_maps["pitch_labels"]
ROLL_LABELS = label_maps["roll_labels"]
PITCH_BIN_EDGES = label_maps["pitch_bin_edges"]
ROLL_BIN_EDGES = label_maps["roll_bin_edges"]


def bin_label(value, edges):
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= value < hi or (i == len(edges) - 2 and value == hi):
            return f"{lo}-{hi}"
    return "out_of_range"


# =========================================================================
# MODEL (same architecture as train_model.py / vqa_system.py)
# =========================================================================
class LetterboxResize:
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


model = CockpitNet(len(PITCH_LABELS), len(ROLL_LABELS)).to(DEVICE)
checkpoint = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

if args.use_letterbox:
    print("Using LETTERBOX preprocessing.")
    resize_step = LetterboxResize(args.img_width, args.img_height)
else:
    print("Using PLAIN (stretch) resize — matches the original epoch-16 checkpoint.")
    resize_step = transforms.Resize((args.img_height, args.img_width))

transform = transforms.Compose([
    resize_step,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def infer_phase(alt, spd, pitch):
    """Same heuristic as vqa_system.py — copied here so this evaluation
    stays in sync with what's actually deployed. If you change one,
    change both."""
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


# =========================================================================
# RUN INFERENCE ON THE FULL TEST SET
# =========================================================================
test_df = pd.read_csv(TEST_CSV)
print(f"Evaluating on {len(test_df)} test rows...")

alt_mean, alt_std = norm_stats["altitude"]["mean"], norm_stats["altitude"]["std"]
spd_mean, spd_std = norm_stats["airspeed"]["mean"], norm_stats["airspeed"]["std"]

results = []
with torch.no_grad():
    for i, row in test_df.iterrows():
        img_path = os.path.join(IMAGES_DIR, row["image"])
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(DEVICE)

        pred = model(img_t)
        pred_altitude = pred["altitude"].item() * alt_std + alt_mean
        pred_airspeed = pred["airspeed"].item() * spd_std + spd_mean
        pred_pitch_idx = pred["pitch_logits"].argmax(1).item()
        pred_roll_idx = pred["roll_logits"].argmax(1).item()
        pred_pitch_bin = PITCH_LABELS[pred_pitch_idx]
        pred_roll_bin = ROLL_LABELS[pred_roll_idx]
        pred_pitch_mid = (PITCH_BIN_EDGES[pred_pitch_idx] + PITCH_BIN_EDGES[pred_pitch_idx + 1]) / 2

        true_pitch_bin = bin_label(row["pitch"], PITCH_BIN_EDGES)
        true_roll_bin = bin_label(row["roll"], ROLL_BIN_EDGES)

        pred_phase = infer_phase(pred_altitude, pred_airspeed, pred_pitch_mid)
        true_phase = row["phase"]

        results.append({
            "image": row["image"],
            "true_phase": true_phase,
            "pred_phase": pred_phase,
            "true_pitch_bin": true_pitch_bin,
            "pred_pitch_bin": pred_pitch_bin,
            "true_roll_bin": true_roll_bin,
            "pred_roll_bin": pred_roll_bin,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test_df)} done...")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "full_test_predictions.csv"), index=False)


# =========================================================================
# METRICS + CONFUSION MATRICES
# =========================================================================
def confusion_and_accuracy(df, true_col, pred_col, all_labels, name):
    accuracy = (df[true_col] == df[pred_col]).mean()
    print(f"\n{'=' * 60}")
    print(f"{name} — overall accuracy: {accuracy:.3f} ({(df[true_col] == df[pred_col]).sum()}/{len(df)})")

    cm = pd.crosstab(df[true_col], df[pred_col], rownames=["true"], colnames=["pred"])
    # Ensure all labels appear as rows/cols even if a class had 0 predictions
    cm = cm.reindex(index=all_labels, columns=all_labels, fill_value=0)
    print(cm.to_string())

    # Per-class recall (row-normalized) — identifies weak classes specifically
    row_sums = cm.sum(axis=1)
    recall = (np.diag(cm) / row_sums.replace(0, np.nan)).fillna(0)
    print(f"\nPer-class recall for {name}:")
    for label, r in recall.items():
        n = int(row_sums[label])
        flag = "  <-- WEAK" if r < 0.6 and n > 0 else ""
        print(f"  {label}: {r:.3f} (n={n}){flag}")

    cm.to_csv(os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_confusion.csv"))

    # Save a heatmap image
    fig, ax = plt.subplots(figsize=(max(5, len(all_labels) * 0.8), max(4, len(all_labels) * 0.7)))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(all_labels)))
    ax.set_yticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_yticklabels(all_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} Confusion Matrix (accuracy={accuracy:.3f})")
    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            val = cm.values[i, j]
            color = "white" if val > cm.values.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_confusion.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap: {fig_path}")

    return accuracy, cm, recall


PHASE_LABELS = ["engine_off", "taxi", "takeoff_roll", "climb", "level_flight"]
phase_acc, phase_cm, phase_recall = confusion_and_accuracy(
    results_df, "true_phase", "pred_phase", PHASE_LABELS, "Phase Inference")

pitch_acc, pitch_cm, pitch_recall = confusion_and_accuracy(
    results_df, "true_pitch_bin", "pred_pitch_bin", PITCH_LABELS, "Pitch Bin")

roll_acc, roll_cm, roll_recall = confusion_and_accuracy(
    results_df, "true_roll_bin", "pred_roll_bin", ROLL_LABELS, "Roll Bin")

print(f"\n{'=' * 60}")
print("SUMMARY (for report/limitations section)")
print(f"{'=' * 60}")
print(f"Phase inference accuracy (full test set, n={len(results_df)}): {phase_acc:.3f}")
print(f"Pitch bin classification accuracy: {pitch_acc:.3f}")
print(f"Roll bin classification accuracy: {roll_acc:.3f}")
print(f"\nAll confusion matrices (CSV + PNG) and per-image predictions saved to:")
print(f"  {OUTPUT_DIR}")
