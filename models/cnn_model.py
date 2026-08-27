"""
CNN VQA Model — ResNet50 + BiGRU + Cross-Attention
No CLIP, no embeddings. Pure pixel reading.

Design for 70%+ accuracy:
- ResNet50 pretrained backbone (fine-tune last 2 layers)
- Bidirectional GRU for questions
- Multi-head cross-attention
- Deep 3-layer classifier
- Dropout regularisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Tuple


class CNNVisualEncoder(nn.Module):
    """
    ResNet50 backbone.
    Unfreeze last 2 layers (layer3, layer4) for fine-tuning
    so the CNN learns radar-specific visual features.
    """
    def __init__(self, out_dim: int = 512):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze early layers (layer1, layer2)
        for name, param in backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Remove final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj     = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(0.3))

    def forward(self, x):
        return self.proj(self.features(x))


class QuestionEncoder(nn.Module):
    """Character-level bidirectional GRU."""
    def __init__(self, vocab: int = 40, embed: int = 128,
                 hidden: int = 512, out_dim: int = 512,
                 num_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab+1, embed, padding_idx=0)
        self.gru   = nn.GRU(embed, hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.3,
                            bidirectional=True)
        self.proj  = nn.Sequential(
            nn.Linear(hidden*2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(0.3))

    def forward(self, x):
        emb  = self.embed(x)
        _, h = self.gru(emb)
        # Concat last forward + backward hidden state
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(h_cat)


class CNNVQAModel(nn.Module):
    """
    Full CNN VQA model.
    Pipeline:
      raw pixels (224x224)
           ↓
      ResNet50 (partially fine-tuned)
           ↓
      512-d visual features
           ↓
      Cross-attention with GRU question features
           ↓
      3-layer classifier → answer
    """
    def __init__(self, num_classes: int, device: str = "cpu",
                 feat_dim: int = 512):
        super().__init__()
        self.device       = device
        self.visual_enc   = CNNVisualEncoder(out_dim=feat_dim)
        self.question_enc = QuestionEncoder(out_dim=feat_dim)

        # Cross-attention: visual attends to question
        self.cross_attn = nn.MultiheadAttention(
            feat_dim, num_heads=8, dropout=0.2, batch_first=True)
        self.norm1 = nn.LayerNorm(feat_dim)

        # Self-attention on combined features
        self.self_attn = nn.MultiheadAttention(
            feat_dim, num_heads=8, dropout=0.2, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)

        # Deep classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.BatchNorm1d(feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim // 2, num_classes))
        self.to(device)

    def forward(self, images: torch.Tensor,
                questions: torch.Tensor) -> torch.Tensor:
        images    = images.to(self.device)
        questions = questions.to(self.device)

        # Extract features
        v = self.visual_enc(images)        # (B, 512)
        q = self.question_enc(questions)   # (B, 512)

        # Cross-attention: visual queries question
        v_s = v.unsqueeze(1)
        q_s = q.unsqueeze(1)
        att, _ = self.cross_attn(v_s, q_s, q_s)
        fused   = self.norm1(v_s + att).squeeze(1)

        # Concatenate and classify
        combined = torch.cat([fused, q], dim=-1)  # (B, 1024)
        return self.classifier(combined)

    def predict(self, images, questions,
                answer_index: Dict) -> Tuple[List[str], List[float]]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(images, questions)
            probs  = F.softmax(logits, dim=-1)
            ids    = logits.argmax(-1).cpu().tolist()
            confs  = probs.max(-1).values.cpu().tolist()
        idx2ans = {v: k for k, v in answer_index.items()}
        return [idx2ans.get(i, "unknown") for i in ids], confs

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(
            torch.load(path, map_location=self.device,
                       weights_only=True))
        return self
