"""CNN VQA Evaluator — accuracy, F1, confusion matrix."""

import os, json, logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from typing import Dict, List
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report,
                             confusion_matrix)

log = logging.getLogger(__name__)


class CNNEvaluator:
    def __init__(self, engine, test_json: str,
                 answer_index: Dict,
                 output_dir: str = "evaluation_results"):
        self.engine       = engine
        self.answer_index = answer_index
        self.output_dir   = output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(test_json) as f:
            self.records = json.load(f)
        log.info(f"Evaluator: {len(self.records)} test records.")

    def run(self) -> Dict:
        y_true, y_pred = [], []
        for rec in tqdm(self.records, desc="Evaluating"):
            try:
                img = Image.open(rec["image_path"]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (0, 26, 0))
            res  = self.engine.answer(img, rec["question"])
            pred = res["answer"].lower().strip()
            gt   = rec["answer"].lower().strip()
            y_true.append(gt)
            y_pred.append(pred)

        labels   = sorted(set(y_true + y_pred))
        overall  = accuracy_score(y_true, y_pred) * 100
        macro_f1 = f1_score(y_true, y_pred, labels=labels,
                            average="macro", zero_division=0)
        report   = classification_report(
            y_true, y_pred, labels=labels,
            zero_division=0, output_dict=True)
        cm       = confusion_matrix(y_true, y_pred, labels=labels)

        metrics = {"overall_acc": round(overall, 2),
                   "macro_f1":    round(macro_f1, 4),
                   "report": report, "cm": cm,
                   "cm_labels": labels,
                   "y_true": y_true, "y_pred": y_pred}

        save = {k: v for k, v in metrics.items()
                if k not in ("cm","y_true","y_pred")}
        save["cm"] = cm.tolist()
        with open(os.path.join(self.output_dir,
                               "cnn_metrics.json"), "w") as f:
            json.dump(save, f, indent=2)

        log.info(f"Overall={overall:.2f}% F1={macro_f1:.4f}")
        return metrics

    def plot_confusion_matrix(self, metrics: Dict,
                              max_classes: int = 30):
        cm, labels = metrics["cm"], metrics["cm_labels"]
        if len(labels) > max_classes:
            idx    = np.argsort(cm.sum(axis=1))[-max_classes:]
            cm     = cm[np.ix_(idx, idx)]
            labels = [labels[i] for i in idx]
        h = max(6, len(labels) * 0.35)
        fig, ax = plt.subplots(figsize=(h+2, h))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45,
                           ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("CNN Confusion Matrix")
        thresh = cm.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center",
                            fontsize=6,
                            color="white" if cm[i, j] > thresh
                            else "black")
        plt.tight_layout()
        path = os.path.join(self.output_dir,
                            "cnn_confusion_matrix.png")
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.show()
        log.info(f"Confusion matrix → {path}")

    def print_report(self, metrics: Dict):
        rep = metrics["report"]
        print(f"\n{'='*62}")
        print(f"{'CLASS':<28} {'PREC':>6} {'REC':>6} "
              f"{'F1':>6} {'SUP':>6}")
        print(f"{'-'*62}")
        for cls, v in rep.items():
            if cls in ("accuracy","macro avg","weighted avg"):
                continue
            if isinstance(v, dict):
                print(f"{cls:<28} {v['precision']:>6.3f} "
                      f"{v['recall']:>6.3f} "
                      f"{v['f1-score']:>6.3f} "
                      f"{int(v['support']):>6}")
        ma = rep.get("macro avg", {})
        print(f"{'-'*62}")
        print(f"{'macro avg':<28} "
              f"{ma.get('precision',0):>6.3f} "
              f"{ma.get('recall',0):>6.3f} "
              f"{ma.get('f1-score',0):>6.3f}")
        print(f"{'='*62}\n")
