"""
capture_phase_synthetic.py
For use with --fdm=null (physics disabled, so properties don't self-update).
Directly varies pitch, altitude, heading, airspeed, roll across frames to
build realistic variation for climb / level_flight phases without fighting
physics. Saves full-panel images only, with exact telemetry (plus pitch/roll
bin labels for classification heads) logged to flight_log.csv.
"""

import argparse
import requests
import time
import csv
import os
import random
import winsound
from mss import mss
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--phase", required=True, choices=["takeoff_roll", "climb", "level_flight"])
parser.add_argument("--frames", type=int, default=60)
parser.add_argument("--headings", type=int, nargs="+", default=[0, 90, 180, 270])
parser.add_argument("--rotate_every", type=int, default=15)

# Climb defaults: altitude ramps up across frames, pitch varies 5-15
parser.add_argument("--alt_start", type=float, default=500)
parser.add_argument("--alt_end", type=float, default=3000)
parser.add_argument("--pitch_min", type=float, default=5)
parser.add_argument("--pitch_max", type=float, default=15)
parser.add_argument("--speed_min", type=float, default=60)
parser.add_argument("--speed_max", type=float, default=90)

parser.add_argument("--roll_min", type=float, default=-10.0)
parser.add_argument("--roll_max", type=float, default=10.0)

parser.add_argument("--start_frame", type=int, default=0,
                     help="Resume from this frame index instead of 0 — use after a crash "
                          "to continue without losing already-captured frames. Opens CSVs "
                          "in append mode when > 0.")

args = parser.parse_args()

base_url = "http://localhost:5500/json"
session = requests.Session()

props = {
    "altitude": "/position/altitude-ft",
    "pitch": "/orientation/pitch-deg",
    "roll": "/orientation/roll-deg",
    "heading": "/orientation/heading-deg",
    "airspeed": "/velocities/airspeed-kt",
    "latitude": "/position/latitude-deg",
    "longitude": "/position/longitude-deg",
    "vertical_speed": "/velocities/vertical-speed-fps",
    "throttle": "/controls/engines/engine/throttle",
    "flaps": "/controls/flight/flaps",
    "gear": "/gear/gear/position-norm",
}

base_folder = r"C:\Users\msasr\OneDrive\Desktop\cap_data"
image_folder = os.path.join(base_folder, args.phase, "images")
gauges_folder = os.path.join(base_folder, args.phase, "gauges")
csv_path = os.path.join(base_folder, args.phase, "flight_log.csv")
os.makedirs(image_folder, exist_ok=True)

# Pixel boxes calibrated against the native 1918x1198 FlightGear window
# (this monitor's native res — --geometry above 1920x1200 is capped by
# the display and won't render larger). "attitude" is a TIGHT crop on
# just the moving horizon bar, not the full gauge bezel, since the full
# bezel crop measured only ~1.6/255 mean pixel difference between
# meaningfully different pitch/roll values — too weak a signal to learn.
GAUGE_BOXES = {
    "airspeed":   (720, 570, 880, 730),
    "attitude":   (900, 590, 1010, 700),   # tight crop on horizon bar only
    "altimeter":  (1030, 570, 1190, 730),
    "turn_coord": (720, 745, 880, 900),
    "heading":    (875, 745, 1030, 900),
    "vsi":        (1030, 745, 1190, 900),
}
# Which telemetry field each gauge crop is labeled with.
# latitude/longitude intentionally excluded — not visually present on
# this panel, so they stay in flight_log.csv as metadata only, never
# as a gauge-model training target.
GAUGE_TO_FIELD = {
    "airspeed": "airspeed",
    "attitude": "pitch",
    "altimeter": "altitude",
    "turn_coord": "roll",
    "heading": "heading",
    "vsi": "vertical_speed",
}
for gauge_name in GAUGE_BOXES:
    os.makedirs(os.path.join(gauges_folder, gauge_name), exist_ok=True)

# Binned classification edges for pitch/roll (coarser bins, per your choice) —
# these are logged as extra columns in flight_log.csv so you can train
# classification heads for pitch/roll without a separate label file.
PITCH_BIN_EDGES = [5, 7, 9, 11, 13, 15]   # 5 bins, 2 degree width
ROLL_BIN_EDGES = [-10, -6, -2, 2, 6, 10]  # 5 bins, 4 degree width


def bin_label(value, edges):
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= value < hi or (i == len(edges) - 2 and value == hi):
            return f"{lo}-{hi}"
    return "out_of_range"


def set_prop(path, value):
    session.post(base_url + path, json={"value": value}, timeout=6.0)


def get_prop(path):
    r = session.get(base_url + path, timeout=6.0)
    return float(r.json()["value"])


def set_prop_verified(path, value, tolerance=0.5, max_retries=3, settle=0.4):
    """Set a property, read it back, retry if it didn't land within tolerance.
    Returns the final confirmed value (may differ slightly from requested)."""
    actual = value
    for attempt in range(max_retries):
        set_prop(path, value)
        time.sleep(settle)
        actual = get_prop(path)
        if abs(actual - value) <= tolerance:
            return actual
        print(f"    retry {attempt + 1}: {path} wanted {value:.2f}, got {actual:.2f}")
    return actual


print("=" * 60)
print(f"SYNTHETIC CAPTURE: {args.phase.upper()}  (fdm=null mode)")
print(f"Frames: {args.frames}")
print("=" * 60)
time.sleep(2)

csv_mode = "a" if args.start_frame > 0 else "w"
write_header = args.start_frame == 0

with open(csv_path, csv_mode, newline="") as f, \
     open(os.path.join(base_folder, args.phase, "gauge_labels.csv"), csv_mode, newline="") as gf:
    writer = csv.writer(f)
    gauge_writer = csv.writer(gf)
    if write_header:
        writer.writerow(["image", "time", "phase"] + list(props.keys()) + ["pitch_bin", "roll_bin"])
        gauge_writer.writerow(["gauge_image", "gauge_type", "label_field", "label_value", "source_image"])
    else:
        print(f"Resuming from frame {args.start_frame} — appending to existing CSVs, not overwriting.")

    start_time = time.time()

    # Build a balanced pitch x roll grid: every bin combination gets
    # (approximately) equal representation, instead of independent random
    # sampling which can over/under-sample combinations by chance.
    if args.phase == "climb":
        pitch_bin_ranges = list(zip(PITCH_BIN_EDGES[:-1], PITCH_BIN_EDGES[1:]))
        roll_bin_ranges = list(zip(ROLL_BIN_EDGES[:-1], ROLL_BIN_EDGES[1:]))
        combos = [(p, r) for p in pitch_bin_ranges for r in roll_bin_ranges]
        random.shuffle(combos)
        # Repeat the combo list enough times to cover all requested frames,
        # then trim to exactly args.frames — guarantees near-equal counts
        # per bin combination rather than relying on random luck.
        reps_needed = (args.frames // len(combos)) + 1
        schedule = (combos * reps_needed)[:args.frames]
        random.shuffle(schedule)  # so bin order isn't sequential across frames
        print(f"Balanced grid: {len(combos)} pitch x roll bin combinations, "
              f"~{args.frames / len(combos):.1f} samples each")

    with mss() as sct:
        # Capture the FULL FlightGear window at native resolution
        # (1918x1198 on this monitor). Both the panel-only crop and
        # the gauge crops are derived from this same raw screenshot,
        # so their coordinates stay consistent with each other.
        monitor = {"top": 0, "left": 0, "width": 1918, "height": 1198}
        # Panel-only region within the raw window (excludes menu bar,
        # sky/windshield, and taskbar) — used for the full-panel training image
        PANEL_BOX = (0, 560, 1918, 1130)

        for count in range(args.start_frame, args.frames):
            frac = count / max(1, args.frames - 1)

            if args.phase == "climb":
                altitude = args.alt_start + frac * (args.alt_end - args.alt_start)
                pitch_range, roll_range = schedule[count]
                # Jitter within the assigned bin, so values aren't all
                # exactly at bin edges/centers — keeps continuous variety
                # for the regression targets while guaranteeing bin coverage
                pitch = random.uniform(pitch_range[0], pitch_range[1])
                roll = random.uniform(roll_range[0], roll_range[1])
                speed = random.uniform(args.speed_min, args.speed_max)
            elif args.phase == "takeoff_roll":
                # Wheels still on the ground: altitude stays near field
                # elevation, speed ramps up with the roll (accelerating
                # down the runway), pitch/roll stay small since the plane
                # hasn't rotated or banked yet.
                altitude = args.alt_start + frac * (args.alt_end - args.alt_start)
                speed = args.speed_min + frac * (args.speed_max - args.speed_min) \
                    + random.uniform(-3, 3)  # small jitter so it's not a perfect ramp
                pitch = random.uniform(0, 5)
                roll = random.uniform(-2, 2)
            else:  # level_flight
                altitude = args.alt_start + random.uniform(-50, 50)
                pitch = random.uniform(-2, 2)
                roll = random.uniform(args.roll_min, args.roll_max)
                speed = random.uniform(args.speed_min, args.speed_max)

            heading_idx = (count // args.rotate_every) % len(args.headings) if args.rotate_every > 0 else 0
            heading = args.headings[heading_idx]

            print(f"[{count + 1}/{args.frames}] Setting alt={altitude:.0f} pitch={pitch:.1f} "
                  f"roll={roll:.1f} hdg={heading} spd={speed:.0f}...")

            frame_ok = False
            for attempt in range(3):
                try:
                    confirmed = {}
                    confirmed["altitude"] = set_prop_verified("/position/altitude-ft", altitude, tolerance=2.0)
                    confirmed["pitch"] = set_prop_verified("/orientation/pitch-deg", pitch, tolerance=0.3)
                    confirmed["roll"] = set_prop_verified("/orientation/roll-deg", roll, tolerance=0.3)
                    confirmed["heading"] = set_prop_verified("/orientation/heading-deg", heading, tolerance=0.5)
                    confirmed["airspeed"] = set_prop_verified("/velocities/airspeed-kt", speed, tolerance=1.0)

                    time.sleep(0.5)

                    data = {}
                    for key, path in props.items():
                        if key in confirmed:
                            continue
                        try:
                            data[key] = get_prop(path)
                        except Exception:
                            data[key] = 0.0
                    data.update(confirmed)

                    # Re-check confirmed values immediately before the screenshot,
                    # so the CSV row matches exactly what's rendered
                    data["altitude"] = get_prop("/position/altitude-ft")
                    data["pitch"] = get_prop("/orientation/pitch-deg")
                    data["roll"] = get_prop("/orientation/roll-deg")
                    data["heading"] = get_prop("/orientation/heading-deg")
                    data["airspeed"] = get_prop("/velocities/airspeed-kt")

                    image_name = f"{args.phase}_{count:04d}.png"
                    image_path = os.path.join(image_folder, image_name)
                    sct_img = sct.grab(monitor)
                    raw_img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)

                    # Full-panel training image: crop out menu bar / sky / taskbar
                    panel_img = raw_img.crop(PANEL_BOX)
                    panel_img.save(image_path)

                    # Verify nothing drifted between pre-screenshot check and the shot itself
                    post_roll = get_prop("/orientation/roll-deg")
                    post_pitch = get_prop("/orientation/pitch-deg")
                    if abs(post_roll - data["roll"]) > 1.0 or abs(post_pitch - data["pitch"]) > 1.0:
                        print(f"  WARNING: value drifted during screenshot (roll {data['roll']:.1f}->{post_roll:.1f}, "
                              f"pitch {data['pitch']:.1f}->{post_pitch:.1f}) — logging actual post-shot values")
                        data["roll"] = post_roll
                        data["pitch"] = post_pitch

                    # Per-gauge crops for the gauge-level training pipeline.
                    # GAUGE_BOXES coordinates are absolute within the raw (uncropped)
                    # window screenshot, matching the calibration screenshot.
                    for gauge_name, box in GAUGE_BOXES.items():
                        gauge_img = raw_img.crop(box)
                        gauge_image_name = f"{args.phase}_{count:04d}_{gauge_name}.png"
                        gauge_img.save(os.path.join(gauges_folder, gauge_name, gauge_image_name))
                        field = GAUGE_TO_FIELD[gauge_name]
                        gauge_writer.writerow([gauge_image_name, gauge_name, field,
                                                round(data[field], 3), image_name])

                    pitch_bin = bin_label(data["pitch"], PITCH_BIN_EDGES)
                    roll_bin = bin_label(data["roll"], ROLL_BIN_EDGES)

                    t = round(time.time() - start_time, 2)
                    row = [image_name, t, args.phase] + [round(data[k], 3) for k in props] + [pitch_bin, roll_bin]
                    writer.writerow(row)
                    f.flush()
                    gf.flush()

                    print(f"  Saved {image_name} | confirmed alt={data['altitude']:.0f} pitch={data['pitch']:.1f} "
                          f"({pitch_bin}) roll={data['roll']:.1f} ({roll_bin}) hdg={data['heading']:.0f} "
                          f"spd={data['airspeed']:.0f}")
                    frame_ok = True
                    break

                except requests.exceptions.RequestException as e:
                    print(f"  Network hiccup on attempt {attempt + 1}/3: {e}")
                    print(f"  If this is FlightGear stalling, waiting 5s before retry...")
                    time.sleep(5)

            if not frame_ok:
                print(f"  FAILED after 3 attempts — skipping frame {count}. "
                      f"If this keeps happening, FlightGear may need a restart: "
                      f"resume later with --start_frame {count}")

    print("\nBEEP! Capture complete!")
    for _ in range(3):
        winsound.Beep(1000, 500)
        time.sleep(0.2)

print(f"\nDONE! Captured {args.frames} frames for {args.phase}")
print(f"Images: {image_folder}")
print(f"Logs: {csv_path}")
