import torch
from PIL import Image

from src.models.radar_vqa_model import RadarVQAModel
from src.inference.ocr_inference import predict_callsign

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RadarVQAModel(pretrained=False)

checkpoint = torch.load(
    "models/vqa/best.pt",
    map_location="cpu",
    weights_only=False,
)

state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict, strict=True)
model.to(device).eval()

crop_path = "data/ocr_crops/test/radar_00002_01.png"
crop = Image.open(crop_path).convert("RGB")

result = predict_callsign(model, crop)

print("Predicted callsign:", result["callsign"])
print("Mean character confidence:", f"{result['confidence'] * 100:.2f}%")
print("Character confidences:", result["character_confidences"])