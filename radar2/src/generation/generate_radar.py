
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "synthetic_complete"
IMAGE_DIR = DATA_DIR / "images"
ANNOTATION_DIR = DATA_DIR / "annotations"
PREVIEW_DIR = DATA_DIR / "previews"

MANIFEST_PATH = DATA_DIR / "all_frames.json"

IMAGE_SIZE = 1280
RANDOM_SEED = 20261030

# Exactly 500 images per count category.
COUNT_QUOTAS = {
    0: 500,
    1: 500,
    2: 500,
    3: 500,
    4: 500,
    5: 500,
    6: 500,
    7: 500,
    "8+": 500,
}

MIN_AIRCRAFT = 8
MAX_AIRCRAFT = 12

TOTAL_IMAGES = sum(COUNT_QUOTAS.values())

PNG_COMPRESS_LEVEL = 1

BACKGROUND = (2, 12, 17)
RING_COLOR = (16, 76, 62)
GRID_COLOR = (12, 48, 42)
AIRCRAFT_COLOR = (80, 255, 180)
AIRCRAFT_FILL = (5, 55, 43)
LABEL_COLOR = (90, 255, 185)
CYAN = (60, 220, 235)
MUTED = (50, 135, 112)
LABEL_BACKGROUND = (2, 20, 24)

# Removed the duplicate ACA715 entry.
CALLSIGNS = [
    "AAL204", "UAL517", "BAW921", "DLH430",
    "UAE615", "AFR302", "QTR811", "SIA228",
    "IGO452", "AXB709", "KLM315", "SWR640",
    "THY182", "JAL508", "VIR227", "FDX901",
    "UPS438", "ACA620", "ANA715", "ETD330",
    "QFA125", "RYR602", "EZY418",
    "GIA302", "KAL511", "CPA810", "JBU204",
    "SAS620", "TAP415", "WJA708", "DAL901"
]

FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/lucon.ttf"
]


# ============================================================
# FONTS
# ============================================================

def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)

    return ImageFont.load_default()


FONT_TITLE = load_font(27)
FONT_LABEL = load_font(20)
FONT_SMALL = load_font(16)


# ============================================================
# HASHING / DUPLICATE DETECTION
# ============================================================

def image_hash(image):
    """SHA-256 hash of the actual rendered RGB pixels."""
    return hashlib.sha256(image.tobytes()).hexdigest()


def scene_hash(contacts, fingerprint):
    """
    Include the rendered image fingerprint in the scene hash.

    This is important for zero-aircraft scenes: their contact lists
    are empty, but their radar displays must still be distinct.
    """
    scene = {
        "contacts": [
            (
                c["callsign"],
                c["altitude"],
                c["heading"],
                c["x"],
                c["y"]
            )
            for c in contacts
        ],
        "image_sha256": fingerprint
    }

    payload = json.dumps(scene, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ============================================================
# RADAR BACKGROUND
# ============================================================

def draw_background(draw, rng):
    cx = IMAGE_SIZE // 2
    cy = IMAGE_SIZE // 2 + 15
    max_radius = 490

    for radius in [100, 200, 300, 400, 490]:
        draw.ellipse(
            (cx - radius, cy - radius,
             cx + radius, cy + radius),
            outline=RING_COLOR,
            width=2
        )

    draw.line(
        (cx - max_radius, cy, cx + max_radius, cy),
        fill=GRID_COLOR,
        width=2
    )

    draw.line(
        (cx, cy - max_radius, cx, cy + max_radius),
        fill=GRID_COLOR,
        width=2
    )

    for angle in [45, 135, 225, 315]:
        rad = math.radians(angle)
        x = cx + max_radius * math.cos(rad)
        y = cy + max_radius * math.sin(rad)

        draw.line(
            (cx, cy, x, y),
            fill=GRID_COLOR,
            width=1
        )

    for radius, label in zip(
        [100, 200, 300, 400, 490],
        ["20 NM", "40 NM", "60 NM", "80 NM", "100 NM"]
    ):
        draw.text(
            (cx + radius - 5, cy + 5),
            label,
            font=FONT_SMALL,
            fill=MUTED
        )

    angle = rng.uniform(0, 360)
    rad = math.radians(angle)

    sx = cx + max_radius * math.cos(rad)
    sy = cy + max_radius * math.sin(rad)

    draw.line(
        (cx, cy, sx, sy),
        fill=(25, 125, 95),
        width=3
    )

    draw.ellipse(
        (cx - 6, cy - 6, cx + 6, cy + 6),
        fill=AIRCRAFT_COLOR
    )

    return cx, cy


# ============================================================
# SCREEN HEADER AND FOOTER
# ============================================================

def draw_header(draw, index, rng):
    draw.text(
        (40, 25),
        "TACTICAL AIR SURVEILLANCE SYSTEM",
        font=FONT_TITLE,
        fill=AIRCRAFT_COLOR
    )

    draw.text(
        (40, 65),
        f"RADAR STATION: RDR-{rng.randint(100, 999)}",
        font=FONT_LABEL,
        fill=LABEL_COLOR
    )

    draw.text(
        (40, 98),
        f"DISPLAY ID: SYN-{index:05d}",
        font=FONT_SMALL,
        fill=MUTED
    )

    draw.text(
        (IMAGE_SIZE - 320, 35),
        "MODE: SURVEILLANCE",
        font=FONT_LABEL,
        fill=CYAN
    )

    draw.text(
        (IMAGE_SIZE - 320, 70),
        "STATUS: OPERATIONAL",
        font=FONT_SMALL,
        fill=AIRCRAFT_COLOR
    )


def draw_footer(draw, count, rng):
    y = IMAGE_SIZE - 65

    draw.line(
        (40, y - 15, IMAGE_SIZE - 40, y - 15),
        fill=RING_COLOR,
        width=2
    )

    draw.text(
        (45, y),
        f"TRACKS: {count:02d}",
        font=FONT_LABEL,
        fill=AIRCRAFT_COLOR
    )

    draw.text(
        (300, y),
        f"RANGE: {rng.choice([50, 75, 100, 150])} NM",
        font=FONT_LABEL,
        fill=LABEL_COLOR
    )

    draw.text(
        (650, y + 3),
        "SYNTHETIC SURVEILLANCE DISPLAY",
        font=FONT_SMALL,
        fill=MUTED
    )


# ============================================================
# AIRCRAFT SYMBOL
# ============================================================

def draw_aircraft(draw, x, y, heading, scale):
    angle = math.radians(heading)

    points = [
        (0, -22),
        (-5, 5),
        (-19, 15),
        (-18, 20),
        (-4, 15),
        (-3, 25),
        (3, 25),
        (4, 15),
        (18, 20),
        (19, 15),
        (5, 5)
    ]

    rotated = []

    for px, py in points:
        px *= scale
        py *= scale

        rx = px * math.cos(angle) - py * math.sin(angle)
        ry = px * math.sin(angle) + py * math.cos(angle)

        rotated.append((x + rx, y + ry))

    draw.polygon(
        rotated,
        fill=AIRCRAFT_FILL,
        outline=AIRCRAFT_COLOR
    )

    draw.line(
        rotated + [rotated[0]],
        fill=AIRCRAFT_COLOR,
        width=2
    )

    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]

    return [
        int(min(xs)),
        int(min(ys)),
        int(max(xs)),
        int(max(ys))
    ]


# ============================================================
# CONTACT GENERATION
# ============================================================

def create_contacts(rng, cx, cy, requested_count):
    """
    Generate exactly the requested number of contacts.

    Unlike the old implementation, this function never silently
    skips an aircraft if placement fails.
    """

    contacts = []
    used_callsigns = set()
    used_positions = []

    for contact_number in range(requested_count):

        placed = False

        for attempt in range(1000):
            angle = rng.uniform(0, 2 * math.pi)
            radius = rng.uniform(110, 425)

            x = int(cx + radius * math.cos(angle))
            y = int(cy + radius * math.sin(angle))

            if all(
                math.dist((x, y), position) > 100
                for position in used_positions
            ):
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place contact {contact_number + 1} "
                f"of {requested_count}. Image generation stopped "
                "instead of silently creating an incorrect count."
            )

        available = [
            name for name in CALLSIGNS
            if name not in used_callsigns
        ]

        if not available:
            raise RuntimeError("Not enough unique callsigns.")

        callsign = rng.choice(available)
        used_callsigns.add(callsign)
        used_positions.append((x, y))

        altitude = rng.randrange(10000, 49001, 100)
        heading = rng.randint(0, 359)

        contacts.append({
            "callsign": callsign,
            "altitude": altitude,
            "heading": heading,
            "x": x,
            "y": y,
            "scale": rng.uniform(0.9, 1.15)
        })

    if len(contacts) != requested_count:
        raise RuntimeError(
            f"Requested {requested_count} contacts, "
            f"but generated {len(contacts)}."
        )

    return contacts


# ============================================================
# LABEL DRAWING
# ============================================================

def draw_contact_label(draw, contact):
    x = contact["x"]
    y = contact["y"]

    lines = [
        contact["callsign"],
        f"ALT {contact['altitude']:05d}",
        f"HDG {contact['heading']:03d}"
    ]

    padding = 8
    line_height = 25

    widths = [
        draw.textbbox((0, 0), line, font=FONT_LABEL)[2]
        for line in lines
    ]

    box_width = max(widths) + 2 * padding
    box_height = 3 * line_height + 2 * padding

    if x < IMAGE_SIZE // 2:
        left = x + 32
    else:
        left = x - box_width - 32

    left = max(20, min(left, IMAGE_SIZE - box_width - 20))

    top = max(
        145,
        min(y - box_height // 2, IMAGE_SIZE - 110 - box_height)
    )

    right = left + box_width
    bottom = top + box_height

    connector_x = left if left > x else right

    draw.line(
        (x, y, connector_x, y),
        fill=MUTED,
        width=2
    )

    draw.rectangle(
        (left, top, right, bottom),
        fill=LABEL_BACKGROUND,
        outline=LABEL_COLOR,
        width=2
    )

    text_boxes = []

    for i, line in enumerate(lines):
        tx = left + padding
        ty = top + padding + i * line_height

        draw.text(
            (tx, ty),
            line,
            font=FONT_LABEL,
            fill=LABEL_COLOR
        )

        bbox = draw.textbbox(
            (tx, ty),
            line,
            font=FONT_LABEL
        )

        text_boxes.append({
            "text": line,
            "bbox_xyxy": list(bbox)
        })

    return {
        "bbox_xyxy": [left, top, right, bottom],
        "text_boxes": text_boxes
    }


# ============================================================
# YOLO ANNOTATIONS
# ============================================================

def to_yolo(bbox):
    x1, y1, x2, y2 = bbox

    return (
        ((x1 + x2) / 2) / IMAGE_SIZE,
        ((y1 + y2) / 2) / IMAGE_SIZE,
        (x2 - x1) / IMAGE_SIZE,
        (y2 - y1) / IMAGE_SIZE
    )



def save_yolo_file(name, contacts):
    """
    YOLO classes:
        0 = aircraft symbol
        1 = callsign text only

    Class 1 uses the tight bounding box of the first
    text line, which contains the callsign.
    """

    lines = []

    for contact in contacts:
        callsign_bbox = contact["text_boxes"][0]["bbox_xyxy"]

        annotations = [
            (0, contact["aircraft_bbox"]),
            (1, callsign_bbox)
        ]

        for class_id, bbox in annotations:
            xc, yc, width, height = to_yolo(bbox)

            lines.append(
                f"{class_id} {xc:.6f} {yc:.6f} "
                f"{width:.6f} {height:.6f}"
            )

    path = ANNOTATION_DIR / f"{name}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")

# ============================================================
# GENERATE ONE RADAR IMAGE
# ============================================================

def generate_image(index, rng, requested_count):
    image = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        BACKGROUND
    )

    draw = ImageDraw.Draw(image)

    cx, cy = draw_background(draw, rng)
    draw_header(draw, index, rng)

    contacts = create_contacts(
        rng,
        cx,
        cy,
        requested_count
    )

    for contact in contacts:
        contact["aircraft_bbox"] = draw_aircraft(
            draw,
            contact["x"],
            contact["y"],
            contact["heading"],
            contact["scale"]
        )

        label = draw_contact_label(draw, contact)

        contact["label_bbox"] = label["bbox_xyxy"]
        contact["text_boxes"] = label["text_boxes"]

    draw_footer(draw, len(contacts), rng)

    metadata_contacts = []

    for contact in contacts:
        metadata_contacts.append({
            "callsign": contact["callsign"],
            "altitude": contact["altitude"],
            "heading": contact["heading"],
            "position": [contact["x"], contact["y"]],
            "aircraft_bbox_xyxy": contact["aircraft_bbox"],
            "label_bbox_xyxy": contact["label_bbox"],
            "text_boxes": contact["text_boxes"]
        })

    metadata = {
        "image": f"radar_{index:05d}.png",
        "width": IMAGE_SIZE,
        "height": IMAGE_SIZE,
        "contact_count": len(contacts),
        "contacts": metadata_contacts
    }

    return image, contacts, metadata


# ============================================================
# BUILD BALANCED COUNT SCHEDULE
# ============================================================

def create_count_schedule(rng):
    """
    Create exactly 500 images for each count category.

    The 8+ category contains a random count from 8 through 12.
    """

    schedule = []

    for count, quota in COUNT_QUOTAS.items():
        for _ in range(quota):
            if count == "8+":
                actual_count = rng.randint(
                    MIN_AIRCRAFT,
                    MAX_AIRCRAFT
                )
            else:
                actual_count = count

            schedule.append(actual_count)

    rng.shuffle(schedule)

    return schedule


# ============================================================
# MAIN DATASET GENERATION
# ============================================================

def main():
    # Never mix this run with existing data.
    for directory in [IMAGE_DIR, ANNOTATION_DIR, PREVIEW_DIR]:
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(
                f"Output directory is not empty:\n{directory}\n\n"
                "This script will not overwrite or mix existing data."
            )

    if MANIFEST_PATH.exists():
        raise RuntimeError(
            f"Manifest already exists:\n{MANIFEST_PATH}\n\n"
            "Move it aside before starting a new run."
        )

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    count_schedule = create_count_schedule(rng)

    seen_image_hashes = set()
    seen_scene_hashes = set()

    manifest = []
    generated_counts = Counter()

    accepted = 0
    attempts = 0
    duplicates_rejected = 0

    start_time = time.time()

    print("=" * 65)
    print("COMPLETE SYNTHETIC RADAR DATASET GENERATION")
    print("=" * 65)
    print(f"Target images : {TOTAL_IMAGES}")
    print(f"Resolution    : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Output        : {IMAGE_DIR}")
    print("\nTarget distribution:")

    for count, quota in COUNT_QUOTAS.items():
        print(f"  {str(count):<5} : {quota}")

    print()

    while accepted < TOTAL_IMAGES:
        attempts += 1

        requested_count = count_schedule[accepted]
        candidate_index = accepted + 1

        image, contacts, metadata = generate_image(
            candidate_index,
            rng,
            requested_count
        )

        if len(contacts) != requested_count:
            raise RuntimeError(
                f"Image {candidate_index}: expected "
                f"{requested_count} aircraft, got {len(contacts)}."
            )

        fingerprint = image_hash(image)
        scene_fingerprint = scene_hash(contacts, fingerprint)

        if (
            fingerprint in seen_image_hashes
            or scene_fingerprint in seen_scene_hashes
        ):
            duplicates_rejected += 1
            continue

        seen_image_hashes.add(fingerprint)
        seen_scene_hashes.add(scene_fingerprint)

        accepted += 1
        generated_counts[len(contacts)] += 1

        name = f"radar_{accepted:05d}"

        metadata["image"] = f"{name}.png"
        metadata["image_sha256"] = fingerprint
        metadata["scene_sha256"] = scene_fingerprint

        image.save(
            IMAGE_DIR / f"{name}.png",
            format="PNG",
            compress_level=PNG_COMPRESS_LEVEL
        )

        save_yolo_file(name, contacts)

        if accepted <= 20:
            image.save(
                PREVIEW_DIR / f"{name}.png",
                format="PNG",
                compress_level=PNG_COMPRESS_LEVEL
            )

        (ANNOTATION_DIR / f"{name}.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8"
        )

        manifest.append(metadata)

        if accepted % 100 == 0 or accepted == 1:
            elapsed = time.time() - start_time
            rate = accepted / max(elapsed, 0.001)
            remaining = TOTAL_IMAGES - accepted
            eta_minutes = remaining / max(rate, 0.001) / 60

            print(
                f"Generated {accepted:04d}/{TOTAL_IMAGES} | "
                f"Duplicates: {duplicates_rejected} | "
                f"Speed: {rate:.1f} images/sec | "
                f"ETA: {eta_minutes:.1f} min"
            )

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 65)
    print("DATASET GENERATION COMPLETED")
    print("=" * 65)

    print(f"Accepted images     : {accepted}")
    print(f"Duplicate rejections: {duplicates_rejected}")
    print(f"Total attempts      : {attempts}")
    print(f"Unique image hashes : {len(seen_image_hashes)}")
    print(f"Unique scene hashes : {len(seen_scene_hashes)}")
    print(f"Time taken          : {elapsed / 60:.2f} minutes")

    print("\nGenerated aircraft-count distribution:")

    for count in range(13):
        print(
            f"  {count:<5}: "
            f"{generated_counts[count]:>5}"
        )

    print("\nOutput locations:")
    print(f"Images           : {IMAGE_DIR}")
    print(f"YOLO annotations : {ANNOTATION_DIR}")
    print(f"Previews         : {PREVIEW_DIR}")
    print(f"Manifest         : {MANIFEST_PATH}")

    print("=" * 65)

    assert accepted == TOTAL_IMAGES
    assert len(seen_image_hashes) == TOTAL_IMAGES
    assert len(seen_scene_hashes) == TOTAL_IMAGES

    # Verify requested quotas.
    for count in range(8):
        assert generated_counts[count] == 500

    assert sum(
        generated_counts[count]
        for count in range(8, 13)
    ) == 500


if __name__ == "__main__":
    main()
