from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.radar_vqa_model import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    CHAR_VOCAB_SIZE,
)


# OCR training used ImageNet normalization on 224x96 crops.
OCR_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Character IDs from radar_vqa_model.py.
ID_TO_CHAR = {
    **{i: chr(ord("A") + i) for i in range(26)},
    **{i + 26: str(i) for i in range(10)},
    40: " ",
    41: "-",
    42: "?",
}


@torch.inference_mode()
def predict_callsign(model, crop: Image.Image, max_length: int = 6):
    """
    Predict a callsign from one YOLO-extracted callsign crop.

    Args:
        model: Loaded RadarVQAModel.
        crop: PIL RGB image, preprocessed to 224x96.
        max_length: Maximum number of callsign characters.

    Returns:
        dict containing predicted callsign and mean character confidence.
    """

    if not isinstance(crop, Image.Image):
        raise TypeError("crop must be a PIL Image")

    crop = crop.convert("RGB")

    if crop.size != (224, 96):
        raise ValueError(
            f"Expected a 224x96 crop, received {crop.size}"
        )

    device = next(model.parameters()).device
    model.eval()

    # [1, 3, 96, 224]
    ocr_tensor = OCR_INFERENCE_TRANSFORM(crop).unsqueeze(0).to(device)

    # Encode the crop once. Output: [1, N, 1024].
    visual_features = model.visual_encoder(ocr_tensor)

    decoder = model.ocr_decoder

    # Initialize attention keys and recurrent states once.
    visual_keys = decoder.visual_key(visual_features)
    global_context = visual_features.mean(dim=1)

    hidden_1 = torch.tanh(
        decoder.initial_state_1(global_context)
    )
    hidden_2 = torch.tanh(
        decoder.initial_state_2(global_context)
    )

    # Start autoregressive decoding with BOS.
    current_token = torch.tensor(
        [BOS_ID],
        dtype=torch.long,
        device=device,
    )

    predicted_chars = []
    confidences = []

    for _ in range(max_length):
        # Embed the current input character.
        character_embedding = decoder.embedding(current_token)

        # Bahdanau spatial attention.
        query = decoder.hidden_query(hidden_1).unsqueeze(1)

        attention_energy = torch.tanh(
            visual_keys + query
        )

        attention_scores = decoder.attention_score(
            attention_energy
        ).squeeze(-1)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        visual_context = torch.bmm(
            attention_weights.unsqueeze(1),
            visual_features,
        ).squeeze(1)

        # Update recurrent decoder states.
        decoder_input = torch.cat(
            [character_embedding, visual_context],
            dim=-1,
        )

        hidden_1 = decoder.decoder_cell_1(
            decoder_input,
            hidden_1,
        )

        hidden_2 = decoder.decoder_cell_2(
            hidden_1,
            hidden_2,
        )

        # Predict the next character.
        logits = decoder.output_layer(hidden_2)
        probabilities = F.softmax(logits, dim=-1)

        confidence, next_token = probabilities.max(dim=-1)

        token_id = int(next_token.item())
        confidence = float(confidence.item())

        if token_id == EOS_ID:
            break

        # Ignore special tokens that aren't callsign characters.
        if token_id in (BOS_ID, PAD_ID):
            current_token = next_token
            continue

        predicted_chars.append(
            ID_TO_CHAR.get(token_id, "?")
        )
        confidences.append(confidence)

        current_token = next_token

    callsign = "".join(predicted_chars)

    mean_confidence = (
        sum(confidences) / len(confidences)
        if confidences else 0.0
    )

    return {
        "callsign": callsign,
        "confidence": mean_confidence,
        "character_confidences": confidences,
    }