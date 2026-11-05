from paddleocr import PaddleOCR
from pathlib import Path

image_path = Path("data/ocr_crops/test/radar_00002_01.png")

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)

results = ocr.predict(str(image_path))

print("\nPADDLEOCR RESULT")
for result in results:
    data = result.json
    print(data)