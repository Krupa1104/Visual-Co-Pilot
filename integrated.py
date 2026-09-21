r"""
integrated.py
Single entry point for the Visual Co-Pilot project, routing to either the
cockpit module (cockpit/gui/vqa_system.py) or the radar module
(radar/radar_cli.py).

Each module is launched as a SUBPROCESS with its working directory set to
its own folder. This is deliberate, not a shortcut: both modules do local,
same-folder imports (cockpit's `from copilot_core import CopilotSystem`,
radar's `from radar_system import RadarSystem`), and both branches define
similarly-named files (e.g. cockpit alone has two separate vqa_system.py
files under gui/ and process/). Importing both into one Python process would
risk sys.path / module-name collisions; running each as its own process
with its own cwd sidesteps that entirely and matches how each module's own
CLI already expects to be run.

This script does NOT re-implement or duplicate either module's argument
parsing — it forwards whatever you pass after the module name straight
through to that module's real CLI, unchanged. Run --help on either module
for its actual arguments; this script does not try to keep that list in
sync.

Usage:
  python integrated.py cockpit --data_dir "C:\...\consolidated" --image "climb_0005.png" --question "how high are we flying"
  python integrated.py radar --base_dir "radar_base" --yolo_root "yolov5" --image "radar_base\data\raw\images\frame_0007.png" --question "how many aircraft are visible on the radar?"

  # see each module's own argument list:
  python integrated.py cockpit --help
  python integrated.py radar --help
"""

import os
import sys
import subprocess
import argparse

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

MODULES = {
    "cockpit": {
        "cwd": os.path.join(REPO_ROOT, "cockpit", "gui"),
        "script": "vqa_system.py",
        "description": "Cockpit instrument VQA (altitude/airspeed/heading/"
                        "banking/climbing/flight phase).",
    },
    "radar": {
        "cwd": os.path.join(REPO_ROOT, "radar"),
        "script": "radar_cli.py",
        "description": "Radar-display VQA (aircraft count/position questions "
                        "over rendered radar frames).",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Route to the cockpit or radar VQA module. Any arguments "
                     "after the module name are forwarded as-is to that "
                     "module's own CLI (vqa_system.py or radar_cli.py).",
        usage="python integrated.py {cockpit,radar} [module-specific args...]",
    )
    parser.add_argument(
        "module",
        choices=MODULES.keys(),
        help="Which module to run: " + ", ".join(
            f"{name} ({cfg['description']})" for name, cfg in MODULES.items()
        ),
    )
    # Everything after the module name is opaque to us and forwarded as-is.
    args, remaining = parser.parse_known_args()

    cfg = MODULES[args.module]
    if not os.path.isdir(cfg["cwd"]):
        print(f"ERROR: expected folder not found: {cfg['cwd']}\n"
              f"Make sure both cockpit/ and radar/ are present in this "
              f"checkout (this script assumes it's run from a branch that "
              f"has both).")
        sys.exit(1)

    script_path = os.path.join(cfg["cwd"], cfg["script"])
    if not os.path.isfile(script_path):
        print(f"ERROR: expected script not found: {script_path}")
        sys.exit(1)

    cmd = [sys.executable, cfg["script"]] + remaining
    print(f"Running: {' '.join(cmd)}\n  (cwd={cfg['cwd']})\n")

    result = subprocess.run(cmd, cwd=cfg["cwd"])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
