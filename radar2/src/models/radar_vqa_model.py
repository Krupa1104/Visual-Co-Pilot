
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50



# ============================================================
# CHARACTER VOCABULARY
# 26 uppercase letters + 10 digits + 7 special tokens = 43
# ============================================================

CHAR_VOCAB_SIZE = 43

PAD_ID = 36
UNK_ID = 37
BOS_ID = 38
EOS_ID = 39
SPACE_ID = 40
HYPHEN_ID = 41
QUESTION_MARK_ID = 42

CHAR_TO_ID = {
    **{chr(ord("A") + i): i for i in range(26)},
    **{str(i): 26 + i for i in range(10)},
    "<PAD>": PAD_ID,
    "<UNK>": UNK_ID,
    "<BOS>": BOS_ID,
    "<EOS>": EOS_ID,
    " ": SPACE_ID,
    "-": HYPHEN_ID,
    "?": QUESTION_MARK_ID,
}


def encode_text(text: str, add_boundaries: bool = False) -> list[int]:
    """Convert text into character IDs."""
    text = text.upper()
    ids = [CHAR_TO_ID.get(char, UNK_ID) for char in text]

    if add_boundaries:
        ids = [BOS_ID] + ids + [EOS_ID]

    return ids



# ============================================================
# RESNET50 VISUAL BACKBONE
# ============================================================


class RadarVisualEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # ResNet50 feature extractor.
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        # Sequential indices:
        # 0 = conv1, 1 = bn1, 4 = layer1, 5 = layer2
        # Freeze the stem and ResNet layers 1 and 2.
        frozen_indices = (0, 1, 4, 5)

        for index in frozen_indices:
            for parameter in self.backbone[index].parameters():
                parameter.requires_grad = False

        self.frozen_indices = frozen_indices

        # ResNet50 produces 2048 feature channels.
        self.visual_projection = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        self.cross_modal_projection = nn.Linear(512, 1024)

    def train(self, mode: bool = True):
        # Set the complete module to the requested mode.
        super().train(mode)

        # Keep frozen blocks in evaluation mode so their
        # BatchNorm running statistics cannot change.
        for index in self.frozen_indices:
            self.backbone[index].eval()

        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = self.visual_projection(features)

        # [B, 512, H, W] -> [B, H*W, 512]
        features = features.flatten(2).transpose(1, 2)

        # [B, H*W, 1024]
        return self.cross_modal_projection(features)


# ============================================================
# CHARACTER-LEVEL QUESTION ENCODER
# ============================================================


class QuestionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=CHAR_VOCAB_SIZE,
            embedding_dim=128,
            padding_idx=PAD_ID,
        )

        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

    
    def forward(
        self,
        question_ids: torch.Tensor,
        question_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        from torch.nn.utils.rnn import (
            pack_padded_sequence,
            pad_packed_sequence,
        )

        if question_lengths is None:
            question_lengths = question_ids.ne(PAD_ID).sum(dim=1)

        if torch.any(question_lengths <= 0):
            raise ValueError("Every question must contain at least one character.")

        embedded = self.embedding(question_ids)

        packed = pack_padded_sequence(
            embedded,
            question_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_encoded, _ = self.bigru(packed)

        encoded, _ = pad_packed_sequence(
            packed_encoded,
            batch_first=True,
            total_length=question_ids.size(1),
        )

        positions = torch.arange(
            question_ids.size(1),
            device=question_ids.device,
        )

        mask = positions.unsqueeze(0) < question_lengths.unsqueeze(1)
        mask_float = mask.unsqueeze(-1).to(encoded.dtype)

        pooled = (encoded * mask_float).sum(dim=1)
        pooled = pooled / mask_float.sum(dim=1).clamp(min=1.0)

        return encoded, pooled


# ============================================================
# CROSS-MODAL ATTENTION
# ============================================================

class CrossModalAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=1024,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )

        self.norm = nn.LayerNorm(1024)

    def forward(
        self,
        question_features: torch.Tensor,
        visual_features: torch.Tensor,
        question_ids: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        attended, _ = self.attention(
            query=question_features,
            key=visual_features,
            value=visual_features,
            key_padding_mask=visual_padding_mask,
            need_weights=False,
        )

        fused_sequence = self.norm(question_features + attended)

        mask = question_ids.ne(PAD_ID).unsqueeze(-1)
        mask_float = mask.to(fused_sequence.dtype)

        fused = (fused_sequence * mask_float).sum(dim=1)
        fused = fused / mask_float.sum(dim=1).clamp(min=1.0)

        return fused, fused_sequence


# ============================================================
# SEPARATE CALLSIGN OCR DECODER
# ============================================================


class CallsignOCRDecoder(nn.Module):
    """
    Autoregressive callsign decoder with Bahdanau-style
    spatial attention over the visual feature grid.
    """

    def __init__(self, character_embedding: nn.Embedding):
        super().__init__()

        self.embedding = character_embedding

        # Visual keys and recurrent query are projected to 512 dimensions.
        self.visual_key = nn.Linear(1024, 512)
        self.hidden_query = nn.Linear(512, 512, bias=False)
        self.attention_score = nn.Linear(512, 1, bias=False)

        # Initialise the two decoder layers from visual context.
        self.initial_state_1 = nn.Linear(1024, 512)
        self.initial_state_2 = nn.Linear(1024, 512)

        # Character embedding (128) + attended visual context (1024).
        self.decoder_cell_1 = nn.GRUCell(
            input_size=128 + 1024,
            hidden_size=512,
        )

        self.decoder_cell_2 = nn.GRUCell(
            input_size=512,
            hidden_size=512,
        )

        self.output_layer = nn.Linear(512, CHAR_VOCAB_SIZE)

    def forward(
        self,
        visual_features: torch.Tensor,
        ocr_input_ids: torch.Tensor,
    ) -> torch.Tensor:

        # visual_features: [B, N, 1024]
        # N represents spatial locations in the detected label crop.
        batch_size, num_locations, _ = visual_features.shape
        sequence_length = ocr_input_ids.size(1)

        # Project visual features once.
        visual_keys = self.visual_key(visual_features)

        # Global context is used only to initialise decoder states.
        global_context = visual_features.mean(dim=1)

        hidden_1 = torch.tanh(
            self.initial_state_1(global_context)
        )

        hidden_2 = torch.tanh(
            self.initial_state_2(global_context)
        )

        outputs = []

        for step in range(sequence_length):

            # Character embedding for the current teacher-forcing input.
            character_ids = ocr_input_ids[:, step]
            character_embedding = self.embedding(character_ids)

            # Spatial attention uses the previous decoder hidden state.
            query = self.hidden_query(hidden_1).unsqueeze(1)

            attention_energy = torch.tanh(
                visual_keys + query
            )

            attention_scores = self.attention_score(
                attention_energy
            ).squeeze(-1)

            attention_weights = torch.softmax(
                attention_scores,
                dim=1,
            )

            # Weighted visual context: [B, 1024].
            visual_context = torch.bmm(
                attention_weights.unsqueeze(1),
                visual_features,
            ).squeeze(1)

            # First recurrent decoder layer.
            decoder_input = torch.cat(
                [character_embedding, visual_context],
                dim=-1,
            )

            hidden_1 = self.decoder_cell_1(
                decoder_input,
                hidden_1,
            )

            # Second recurrent decoder layer.
            hidden_2 = self.decoder_cell_2(
                hidden_1,
                hidden_2,
            )

            # Character prediction for this timestep.
            step_logits = self.output_layer(hidden_2)
            outputs.append(step_logits)

        # [B, T, 41]
        return torch.stack(outputs, dim=1)


# ============================================================
# COMPLETE RADAR VQA MODEL
# ============================================================

class RadarVQAModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.visual_encoder = RadarVisualEncoder(pretrained=pretrained)

        self.question_encoder = QuestionEncoder()

        self.cross_modal_attention = CrossModalAttention()

        # 1024 -> 512 -> 256 -> 28
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 28),
        )

        self.ocr_decoder = CallsignOCRDecoder(
            self.question_encoder.embedding
        )

    
    def forward(
        self,
        images: torch.Tensor,
        question_ids: torch.Tensor,
        question_lengths: Optional[torch.Tensor] = None,
        ocr_input_ids: Optional[torch.Tensor] = None,
        ocr_images: Optional[torch.Tensor] = None,
        aircraft_crops: Optional[torch.Tensor] = None,
        aircraft_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:

        # 1. Extract global features from the complete radar image.
        global_features = self.visual_encoder(images)

        # 2. Extract local features from YOLO-detected aircraft crops.
        visual_features = global_features
        visual_padding_mask = None

        if aircraft_crops is not None:
            # aircraft_crops: [B, K, C, H, W]
            batch_size, num_crops, channels, height, width = (
                aircraft_crops.shape
            )

            flattened_crops = aircraft_crops.reshape(
                batch_size * num_crops,
                channels,
                height,
                width,
            )

            # Reuse the SAME ResNet-50 and its trained weights.
            crop_features = self.visual_encoder(flattened_crops)

            # [B*K, N, 1024] -> [B, K*N, 1024]
            num_locations = crop_features.size(1)

            crop_features = crop_features.reshape(
                batch_size,
                num_crops * num_locations,
                1024,
            )

            # Combine global and local visual features.
            visual_features = torch.cat(
                [global_features, crop_features],
                dim=1,
            )

            # True means that the corresponding crop is padding.
            if aircraft_mask is not None:
                region_mask = aircraft_mask.bool().repeat_interleave(
                    num_locations,
                    dim=1,
                )

                global_mask = torch.zeros(
                    batch_size,
                    global_features.size(1),
                    dtype=torch.bool,
                    device=images.device,
                )

                visual_padding_mask = torch.cat(
                    [global_mask, region_mask],
                    dim=1,
                )

        # 3. Encode the user's question.
        question_features, _ = self.question_encoder(
            question_ids,
            question_lengths,
        )

        # 4. Fuse question, global image and aircraft-region features.
        fused_features, _ = self.cross_modal_attention(
            question_features,
            visual_features,
            question_ids,
            visual_padding_mask,
        )

        # 5. Predict the answer.
        classification_logits = self.classifier(fused_features)
        valid_class_logits = classification_logits[:, :27]

        # 6. Preserve the existing callsign OCR pathway.
        ocr_logits = None

        if ocr_input_ids is not None:
            if ocr_images is not None:
                ocr_visual_features = self.visual_encoder(ocr_images)
            else:
                ocr_visual_features = global_features

            ocr_logits = self.ocr_decoder(
                ocr_visual_features,
                ocr_input_ids,
            )

        return {
            "classification_logits": classification_logits,
            "valid_class_logits": valid_class_logits,
            "ocr_logits": ocr_logits,
        }


if __name__ == "__main__":
    print("Radar VQA architecture file imported successfully.")
    print("Character vocabulary size:", CHAR_VOCAB_SIZE)
    print("Valid classification classes:", 27)
    print("Classifier output neurons:", 28)
