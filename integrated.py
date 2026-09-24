r"""
integrate.py
Primary launch dashboard for the Visual Co-Pilot project. Shows two
selectable cards -- Cockpit and Radar -- each of which launches that
module's own Gradio GUI as a subprocess, then opens it in your browser.

DIRECTORY ASSUMPTIONS (see README section at the bottom of this docstring):
  This script must sit at the repo root, next to cockpit/ and radar/,
  e.g. on the `integrate` branch of Krupa1104/Visual-Co-Pilot:
    Visual-Co-Pilot/
    +-- cockpit/gui/app.py   <- cockpit's Gradio entry point (already existed)
    +-- radar/app.py         <- radar's Gradio entry point (NEW -- see note)
    +-- integrate.py         <- this file

NOTE ON radar/app.py:
  As of the `integrate` branch's original contents, radar/ had no
  standalone GUI script -- only radar/gui/cnn_demo.py, which uses
  ipywidgets and only renders inside a Jupyter/Colab kernel (no subprocess
  or browser to launch). This script assumes radar/app.py has been added
  as a Gradio wrapper around RadarSystem (mirroring cockpit/gui/app.py's
  pattern) so both modules have a real, launchable GUI. If radar/app.py
  isn't present at that path, the Radar card will fail with a clear error
  telling you so, rather than silently doing nothing.

EXECUTION STRATEGY:
  Each module is launched as a SUBPROCESS with cwd set to its own script's
  folder (cockpit/gui/ or radar/), exactly like radar_cli.py's own
  docstring insists on ("run this FROM INSIDE ..."). This is deliberate,
  not a shortcut: both modules do local, same-folder imports
  (`from copilot_core import CopilotSystem`, `from radar_system import
  RadarSystem`), and running each as its own process with its own cwd
  avoids sys.path / module-name collisions entirely (both modules define
  similarly-named files, e.g. cockpit alone has two separate
  vqa_system.py files). This launcher does not import either module's
  code directly for that reason.

  Each module is a Gradio app that starts a local web server -- this
  script waits a few seconds for that server to come up, then opens your
  default browser to it automatically. The launcher window then hides
  itself (not process-killed -- see close_launcher_on_launch) so it's out
  of your way, but stays alive in the background to host the subprocess.

SETUP / PATH ASSUMPTIONS FOR RADAR AND COCKPIT (prompted for at runtime,
not hardcoded -- but listed here so you know what to have ready):
  Cockpit needs:
    --data_dir   your local `consolidated` folder (contains model_output/,
                 train.csv, etc.) -- NOT checked into git (~7.6GB, see
                 radar/README.md's note on datasets; cockpit's data is
                 excluded the same way).
  Radar needs:
    --base_dir   your local `radar_base` folder (contains models/, data/,
                 logs/ -- your local copy of aviation_vqa_output from
                 Drive). Also excluded from git via radar/.gitignore.
    --yolo_root  a local `git clone https://github.com/ultralytics/yolov5`
                 checkout. Defaults to <parent of base_dir>/yolov5 if left
                 blank.
  Neither dataset ships in the repo -- get them from your team / Drive
  before using this launcher, or the respective card will fail with a
  FileNotFoundError from inside CopilotSystem / RadarSystem, not from
  this launcher itself.
"""

import os
import sys
import time
import threading
import subprocess
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GRADIO_URL = "http://127.0.0.1:7860"
SERVER_STARTUP_WAIT_SECONDS = 6  # heuristic; model loading can take longer
                                  # on first run (esp. cockpit's ~3GB LLM
                                  # gate) -- if the browser opens too early
                                  # to a blank/refused connection, just
                                  # refresh the tab once loading finishes.

MODULES = {
    "cockpit": {
        "cwd": os.path.join(REPO_ROOT, "cockpit", "gui"),
        "script": "app.py",
        "label": "Cockpit",
        "description": "Instrument VQA: altitude, airspeed, heading, "
                        "banking, climbing, flight phase.",
        "required_paths": [
            ("data_dir", "--data_dir", "Select your consolidated/ data folder", "dir"),
        ],
    },
    "radar": {
        "cwd": os.path.join(REPO_ROOT, "radar"),
        "script": "app.py",
        "label": "Radar",
        "description": "Radar-display VQA: aircraft count, position, "
                        "altitude, callsign.",
        "required_paths": [
            ("base_dir", "--base_dir", "Select your radar_base/ folder (models/, data/, logs/)", "dir"),
            ("yolo_root", "--yolo_root", "Select your local yolov5/ checkout (Cancel = use default)", "dir_optional"),
        ],
    },
}


def prompt_for_paths(module_cfg):
    """Ask the user, via file-picker dialogs, for whatever paths this
    module's app.py needs. Returns a list of CLI args, or None if the
    user cancelled a required one."""
    forwarded = []
    for _name, flag, title, kind in module_cfg["required_paths"]:
        path = filedialog.askdirectory(title=title, mustexist=True)
        if not path:
            if kind == "dir_optional":
                continue  # falls back to that module's own --yolo_root default
            messagebox.showwarning(
                "Required path missing",
                f"{title} is required to launch {module_cfg['label']}.",
            )
            return None
        forwarded += [flag, path]
    return forwarded


def wait_then_open_browser(url, delay_seconds):
    time.sleep(delay_seconds)
    webbrowser.open(url)


def launch_module(module_name, root_window):
    cfg = MODULES[module_name]

    if not os.path.isdir(cfg["cwd"]):
        messagebox.showerror(
            "Folder not found",
            f"Expected folder not found:\n{cfg['cwd']}\n\n"
            f"Make sure both cockpit/ and radar/ are present next to "
            f"integrate.py (this launcher assumes it's run from a branch "
            f"that has both, e.g. 'integrate').",
        )
        return

    script_path = os.path.join(cfg["cwd"], cfg["script"])
    if not os.path.isfile(script_path):
        messagebox.showerror(
            "Entry point not found",
            f"Expected script not found:\n{script_path}\n\n"
            + ("radar/app.py needs to exist as a Gradio wrapper around "
               "RadarSystem -- see this launcher's docstring for why it "
               "isn't part of the original repo contents."
               if module_name == "radar" else
               "cockpit/gui/app.py should already exist -- check you're "
               "on the right branch/checkout."),
        )
        return

    forwarded_args = prompt_for_paths(cfg)
    if forwarded_args is None:
        return  # user cancelled a required path picker

    cmd = [sys.executable, cfg["script"]] + forwarded_args
    try:
        subprocess.Popen(cmd, cwd=cfg["cwd"])
    except OSError as exc:
        messagebox.showerror("Launch failed", f"Could not start {cfg['label']}:\n{exc}")
        return

    messagebox.showinfo(
        "Launching",
        f"Starting {cfg['label']}...\n\n"
        f"This runs: {' '.join(cmd)}\n(cwd={cfg['cwd']})\n\n"
        f"Your browser will open automatically once the server is up "
        f"(~{SERVER_STARTUP_WAIT_SECONDS}s). If it opens to a refused "
        f"connection because model loading took longer than that, just "
        f"refresh the tab.",
    )
    threading.Thread(
        target=wait_then_open_browser,
        args=(DEFAULT_GRADIO_URL, SERVER_STARTUP_WAIT_SECONDS),
        daemon=True,
    ).start()

    # Hide (not destroy) the launcher rather than killing this process --
    # destroying root_window would also kill the daemon thread above before
    # it gets to open the browser, and the subprocess is independent of
    # this process anyway (Popen, not run()), so there's no need to keep
    # this window in the way once a module is launched.
    root_window.withdraw()


def build_gui():
    root = tk.Tk()
    root.title("Visual Co-Pilot -- Launcher")
    root.geometry("520x260")
    root.resizable(False, False)

    tk.Label(root, text="Visual Co-Pilot", font=("Segoe UI", 18, "bold")).pack(pady=(20, 4))
    tk.Label(root, text="Choose a module to launch", font=("Segoe UI", 10)).pack(pady=(0, 16))

    cards_frame = tk.Frame(root)
    cards_frame.pack(expand=True, fill="both", padx=20)

    for module_name in ("cockpit", "radar"):
        cfg = MODULES[module_name]
        card = tk.Frame(cards_frame, relief="groove", borderwidth=2, padx=14, pady=14)
        card.pack(side="left", expand=True, fill="both", padx=10)

        tk.Label(card, text=cfg["label"], font=("Segoe UI", 14, "bold")).pack()
        tk.Label(
            card, text=cfg["description"], wraplength=200, justify="center",
            font=("Segoe UI", 9),
        ).pack(pady=(6, 14))
        tk.Button(
            card, text=f"Launch {cfg['label']}", width=18,
            command=lambda m=module_name: launch_module(m, root),
        ).pack()

    root.mainloop()


if __name__ == "__main__":
    build_gui()