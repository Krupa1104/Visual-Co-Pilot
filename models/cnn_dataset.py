"""
CNN VQA Dataset
Reads raw image pixels. No CLIP, no embeddings.
Questions tokenised as character indices.
"""

import json, torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CNNVQADataset(Dataset):
    CHARS    = list("abcdefghijklmnopqrstuvwxyz0123456789 ?,'.")
    CHAR2IDX = {c: i+1 for i, c in enumerate(CHARS)}
    MAX_Q    = 60

    def __init__(self, json_path: str, split: str = "train",
                 image_size: int = 224):
        with open(json_path) as f:
            self.records = json.load(f)
        self.split = split

        # ImageNet normalisation for ResNet50
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225])

        if split == "train":
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomVerticalFlip(0.1),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(), norm])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(), norm])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Read raw pixels
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 26, 0))
        img_t = self.tf(img)  # (3, 224, 224)

        # Tokenise question as characters
        q     = rec["question"].lower()[:self.MAX_Q]
        q_tok = [self.CHAR2IDX.get(c, 0) for c in q]
        q_tok += [0] * (self.MAX_Q - len(q_tok))
        q_t   = torch.tensor(q_tok, dtype=torch.long)

        label = torch.tensor(rec["label_id"], dtype=torch.long)
        return img_t, q_t, label
