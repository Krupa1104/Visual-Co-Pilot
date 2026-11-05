import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_radar


OUTPUT_DIR = ROOT / "outputs" / "generator_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if any(OUTPUT_DIR.iterdir()):
    raise RuntimeError(
        f"Test folder is not empty: {OUTPUT_DIR}\n"
        "Inspect it before running again."
    )

rng = random.Random(20261030)

cx = generate_radar.IMAGE_SIZE // 2
cy = generate_radar.IMAGE_SIZE // 2 + 15

test_counts = [0, 1, 4, 5, 7, 8, 10, 12]

for index, count in enumerate(test_counts, start=1):
    image, contacts, metadata = generate_radar.generate_image(
        index, rng, count
    )

    output_path = OUTPUT_DIR / f"test_{count:02d}_aircraft.png"
    image.save(output_path)

    with Image.open(output_path) as check:
        assert check.size == (
            generate_radar.IMAGE_SIZE,
            generate_radar.IMAGE_SIZE,
        )

    assert len(contacts) == count

    print(
        f"PASS | {count:2d} aircraft | "
        f"{output_path.name} | {image.size}"
    )

print(f"\nVisual test images saved to:\n{OUTPUT_DIR}")