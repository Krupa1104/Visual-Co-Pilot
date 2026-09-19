"""
Dataset Builder - CNN version (no CLIP).
Reads JSONL, verifies images, builds vocab, 70/30 split.
"""

import os, json, random, logging
from collections import Counter
from typing import Dict, List, Tuple
from PIL import Image

log = logging.getLogger(__name__)
random.seed(42)


class DatasetBuilder:
    def __init__(self, raw_image_dir, annotation_path,
                 processed_dir, train_ratio=0.7, image_size=224):
        self.raw_image_dir   = raw_image_dir
        self.annotation_path = annotation_path
        self.processed_dir   = processed_dir
        self.train_ratio     = train_ratio
        self.image_size      = image_size
        self.img_out         = os.path.join(processed_dir, "images")
        os.makedirs(self.img_out, exist_ok=True)

    def build(self) -> Dict:
        records = self._load()
        records = self._verify_images(records)
        answer_index = self._build_vocab(records)
        train, test  = self._split(records)

        self._save(records,      os.path.join(self.processed_dir, "dataset.json"))
        self._save(train,        os.path.join(self.processed_dir, "train.json"))
        self._save(test,         os.path.join(self.processed_dir, "test.json"))
        self._save(answer_index, os.path.join(self.processed_dir, "answer_index.json"))

        stats = {"total": len(records), "train": len(train),
                 "test":  len(test),    "num_classes": len(answer_index),
                 "top_answers": Counter(r["answer"] for r in records).most_common(10)}
        log.info(f"Dataset: {stats}")
        return stats

    def _load(self):
        records = []
        with open(self.annotation_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        log.info(f"Loaded {len(records)} annotations.")
        return records

    def _verify_images(self, records):
        clean, missing = [], 0
        for rec in records:
            src   = rec.get("image_path", "")
            fname = (os.path.basename(src) if src
                     else f"{rec['image_id']}.png")
            if not os.path.exists(src):
                alt = os.path.join(self.raw_image_dir, fname)
                src = alt if os.path.exists(alt) else ""
            if not src:
                missing += 1
                continue
            dst = os.path.join(self.img_out, fname)
            if not os.path.exists(dst):
                try:
                    Image.open(src).convert("RGB").resize(
                        (self.image_size, self.image_size),
                        Image.LANCZOS).save(dst)
                except Exception as e:
                    log.warning(f"Skip {src}: {e}")
                    missing += 1
                    continue
            rec = dict(rec)
            rec["image_path"] = dst
            clean.append(rec)
        log.info(f"Images: {len(clean)} ok, {missing} missing.")
        return clean

    def _build_vocab(self, records):
        answers = sorted(set(r["answer"] for r in records))
        idx     = {a: i for i, a in enumerate(answers)}
        for rec in records:
            rec["label_id"] = idx[rec["answer"]]
        log.info(f"Vocab: {len(idx)} classes.")
        return idx

    def _split(self, records):
        by_img: Dict[str, List] = {}
        for r in records:
            by_img.setdefault(r["image_id"], []).append(r)
        ids = list(by_img.keys())
        random.shuffle(ids)
        cut = int(len(ids) * self.train_ratio)
        train_ids = set(ids[:cut])
        return ([r for r in records if r["image_id"] in train_ids],
                [r for r in records if r["image_id"] not in train_ids])

    def _save(self, obj, path):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        log.info(f"Saved {path}")
