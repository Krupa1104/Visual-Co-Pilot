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
parsing — beyond the module name itself, it forwards whatever you pass
straight through to that module's real CLI, unchanged. Run --help on
either module for its actual arguments; this script does not try to keep
that list in sync.

Usage (direct, for scripting):
  python integrated.py cockpit --data_dir "C:\...\consolidated" --image "climb_0005.png" --question "how high are we flying"
  python integrated.py radar --base_dir "radar_base" --yolo_root "yolov5" --image "radar_base\data\raw\images\frame_0007.png" --question "how many aircraft are visible on the radar?"

Usage (interactive — just run with no arguments):
  python integrated.py
  # prompts: "Which module? [1] cockpit  [2] radar"
  # then prompts for that module's arguments one at a time

  # see each module's own argument list without running it:
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
        # Prompts shown in interactive mode. (flag, prompt text, required)
        # Kept intentionally minimal — matches vqa_system.py's own
        # --data_dir/--image/--question; everything else uses that
        # script's own defaults. Run `python integrated.py cockpit --help`
        # for the full argument list if you need the rest (img_width,
        # use_letterbox, confidence_threshold, etc.).
        "interactive_prompts": [
            ("--data_dir", "Path to your data/consolidated folder", True),
            ("--image", "Path to the image to run", True),
            ("--question", "Question to ask (leave blank for interactive loop)", False),
        ],
    },
    "radar": {
        "cwd": os.path.join(REPO_ROOT, "radar"),
        "script": "radar_cli.py",
        "description": "Radar-display VQA (aircraft count/position questions "
                        "over rendered radar frames).",
        "interactive_prompts": [
            ("--base_dir", "Local folder holding models/, data/, logs/", True),
            ("--yolo_root", "Path to your yolov5 checkout (blank = default)", False),
            ("--image", "Path to the radar image to run", True),
            ("--question", "Question to ask (leave blank for interactive loop)", False),
        ],
    },
}


def prompt_choose_module():
    names = list(MODULES.keys())
    print("Which module?")
    for i, name in enumerate(names, 1):
        print(f"  [{i}] {name} — {MODULES[name]['description']}")
    while True:
        choice = input(f"Enter 1-{len(names)}: ").strip()
        if choice in [str(i) for i in range(1, len(names) + 1)]:
            return names[int(choice) - 1]
        if choice in MODULES:
            return choice
        print("Not a valid choice, try again.")


def prompt_build_args(module_name):
    cfg = MODULES[module_name]
    forwarded = []
    print(f"\n--- {module_name} arguments (blank = skip/use default) ---")
    for flag, prompt_text, required in cfg["interactive_prompts"]:
        while True:
            value = input(f"{prompt_text}: ").strip()
            if value:
                forwarded += [flag, value]
                break
            if required:
                print(f"  {flag} is required, please enter a value.")
                continue
            break
    return forwarded


def run_module(module_name, forwarded_args):
    cfg = MODULES[module_name]
    if not os.path.isdir(cfg["cwd"]):
        print(f"ERROR: expected folder not found: {cfg['cwd']}\n"
              f"Make sure both cockpit/ and radar/ are present in this "
              f"checkout (this script assumes it's run from a branch that "
              f"has both, e.g. 'integrate').")
        sys.exit(1)

    script_path = os.path.join(cfg["cwd"], cfg["script"])
    if not os.path.isfile(script_path):
        print(f"ERROR: expected script not found: {script_path}")
        sys.exit(1)

    cmd = [sys.executable, cfg["script"]] + forwarded_args
    print(f"\nRunning: {' '.join(cmd)}\n  (cwd={cfg['cwd']})\n")

    result = subprocess.run(cmd, cwd=cfg["cwd"])
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Route to the cockpit or radar VQA module. Run with no "
                     "arguments for an interactive prompt, or pass the "
                     "module name plus its own arguments directly.",
        usage="python integrated.py [cockpit|radar] [module-specific args...]",
    )
    parser.add_argument(
        "module",
        nargs="?",
        choices=MODULES.keys(),
        default=None,
        help="Which module to run. Omit this (and everything else) to be "
             "prompted interactively instead.",
    )
    args, remaining = parser.parse_known_args()

    if args.module is None:
        # Fully interactive: no module name and no args were given at all.
        module_name = prompt_choose_module()
        forwarded = prompt_build_args(module_name)
    else:
        # Direct mode: module named on the command line. If no further args
        # were forwarded either, still offer the interactive prompts for
        # that module's arguments rather than failing on missing required
        # ones — e.g. `python integrated.py cockpit` alone.
        module_name = args.module
        forwarded = remaining if remaining else prompt_build_args(module_name)

    run_module(module_name, forwarded)


if __name__ == "__main__":
    main()