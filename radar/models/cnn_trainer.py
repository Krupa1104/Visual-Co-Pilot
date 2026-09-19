"""
CNN VQA Trainer
Designed to achieve 70%+ validation accuracy.

Key strategies:
- Higher LR for CNN backbone (fine-tuning layer3/layer4)
- Lower LR for new heads
- ReduceLROnPlateau scheduler (reduces LR when val loss stalls)
- Label smoothing 0.1
- Gradient clipping
- Early stopping if val_acc >= 75%
- 20 epochs max
"""

import os, logging, torch, torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class CNNVQATrainer:
    def __init__(self, model, train_loader: DataLoader,
                 val_loader: DataLoader, device: str = "cpu",
                 epochs: int = 20, save_dir: str = "models"):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.epochs       = epochs
        self.save_dir     = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Separate param groups:
        # CNN backbone (fine-tune) gets smaller LR
        # New heads get larger LR
        backbone_params = []
        head_params     = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "visual_enc.features" in name:
                backbone_params.append(p)
            else:
                head_params.append(p)

        self.optim = torch.optim.AdamW([
            {"params": backbone_params, "lr": 5e-5},
            {"params": head_params,     "lr": 3e-4},
        ], weight_decay=1e-2)

        # ReduceLROnPlateau — cuts LR when val loss stops improving
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode="min", factor=0.5,
            patience=3, min_lr=1e-6)

        # Label smoothing helps avoid overconfidence
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def train(self) -> Dict[str, List]:
        history  = {"train_loss":[], "train_acc":[],
                    "val_loss":[],   "val_acc":[]}
        best_acc = 0.0
        best_path = os.path.join(self.save_dir, "cnn_best_model.pth")

        for ep in range(1, self.epochs + 1):
            tl, ta = self._epoch(self.train_loader, train=True)
            vl, va = self._epoch(self.val_loader,   train=False)
            self.scheduler.step(vl)

            history["train_loss"].append(tl)
            history["train_acc"].append(ta)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)

            lr = self.optim.param_groups[1]["lr"]
            log.info(f"Ep {ep:02d}/{self.epochs} | "
                     f"train={ta:.2f}% loss={tl:.4f} | "
                     f"val={va:.2f}% loss={vl:.4f} | lr={lr:.2e}")

            if va > best_acc:
                best_acc = va
                self.model.save(best_path)
                log.info(f"  ✓ Best saved ({va:.2f}%)")

            # Early stop if already exceeding target
            if va >= 75.0:
                log.info(f"  ★ Reached {va:.2f}% — stopping early")
                break

        self.model.save(
            os.path.join(self.save_dir, "cnn_final_model.pth"))
        log.info(f"Training done. Best val acc: {best_acc:.2f}%")
        return history

    def _epoch(self, loader: DataLoader, train: bool):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        for imgs, qs, labels in loader:
            labels = labels.to(self.device)
            if train:
                self.optim.zero_grad()

            logits = self.model(imgs, qs)
            loss   = self.criterion(logits, labels)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters()
                     if p.requires_grad], 1.0)
                self.optim.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(-1) == labels).sum().item()
            total      += labels.size(0)

        return (total_loss / max(total, 1),
                correct / max(total, 1) * 100)

    def plot_history(self, history: Dict[str, List]):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        eps = range(1, len(history["train_loss"]) + 1)
        a1.plot(eps, history["train_loss"], "b-o", label="Train")
        a1.plot(eps, history["val_loss"],   "r-o", label="Val")
        a1.set_title("CNN Loss")
        a1.legend(); a1.grid(alpha=0.3)
        a2.plot(eps, history["train_acc"],  "b-o", label="Train")
        a2.plot(eps, history["val_acc"],    "r-o", label="Val")
        a2.axhline(70, color="green",  linestyle="--",
                   alpha=0.7, label="70% target")
        a2.axhline(75, color="orange", linestyle="--",
                   alpha=0.5, label="75% stretch")
        a2.set_title("CNN Accuracy (%)")
        a2.legend(); a2.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.save_dir, "cnn_training_curves.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.show()
        log.info(f"Curves → {path}")
