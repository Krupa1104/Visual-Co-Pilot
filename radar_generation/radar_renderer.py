"""
Radar Image Renderer — 224x224 PNG PPI radar images.
Strict duplicate detection. Skips already-rendered images.
"""

import os, hashlib, logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Tuple

log = logging.getLogger(__name__)

BG   = "#001a00"
RING = "#00aa33"
BLIP = "#00ff44"
VEC  = "#66ffaa"
LBL  = "#ccffcc"
DPI  = 96
FIG  = 224 / DPI


class RadarRenderer:
    def __init__(self, output_dir="data/raw/images", image_size=224):
        self.output_dir = output_dir
        self.image_size = image_size
        os.makedirs(output_dir, exist_ok=True)
        self._seen = set()

        # Load fingerprints of already-rendered images
        existing = [f for f in os.listdir(output_dir)
                    if f.endswith(".png")]
        log.info(f"Found {len(existing)} existing images in {output_dir}")

    def render_all(self, frames: List[Dict]) -> List[Dict]:
        rendered, skipped = [], 0
        for i, frame in enumerate(frames):
            fp = self._layout_fp(frame["aircraft"])
            if fp in self._seen:
                skipped += 1
                continue
            self._seen.add(fp)

            img_path = os.path.join(self.output_dir,
                                    f"{frame['frame_id']}.png")
            if not os.path.exists(img_path):
                self._render(frame, img_path)

            enriched = dict(frame)
            enriched["image_path"] = img_path
            rendered.append(enriched)

            if (i + 1) % 200 == 0:
                log.info(f"  Rendered {len(rendered)} images …")

        log.info(f"Done: {len(rendered)} rendered, {skipped} skipped.")
        return rendered

    def attach_paths(self, frames: List[Dict],
                     img_dir: str) -> List[Dict]:
        """
        For already-rendered frames: just attach image_path.
        Used when loading existing 2000 frames from Drive.
        """
        out = []
        for fr in frames:
            path = os.path.join(img_dir, f"{fr['frame_id']}.png")
            if os.path.exists(path):
                enriched = dict(fr)
                enriched["image_path"] = path
                out.append(enriched)
                fp = self._layout_fp(fr["aircraft"])
                self._seen.add(fp)
        log.info(f"Attached paths for {len(out)} existing frames.")
        return out

    def _render(self, frame: Dict, save_path: str):
        aircraft = frame["aircraft"]
        norm     = self._normalise(aircraft)

        fig, ax = plt.subplots(figsize=(FIG, FIG), dpi=DPI, facecolor=BG)
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.axis("off")

        for r in [0.15, 0.30, 0.45]:
            ax.add_patch(plt.Circle((0.5, 0.5), r, color=RING,
                                    fill=False, lw=0.6, alpha=0.25))
        ax.plot([0.5,0.5],[0.04,0.96], color=RING, lw=0.4, alpha=0.2)
        ax.plot([0.04,0.96],[0.5,0.5], color=RING, lw=0.4, alpha=0.2)
        ax.plot(0.5, 0.5, "o", color=RING, ms=2, alpha=0.5)

        for ac, (nx, ny) in zip(aircraft, norm):
            ax.add_patch(plt.Circle((nx, ny), 0.014,
                                    color=BLIP, zorder=5))
            hdg = np.deg2rad(ac["heading"])
            vx, vy = np.sin(hdg)*0.05, np.cos(hdg)*0.05
            ax.annotate("", xy=(nx+vx, ny+vy), xytext=(nx, ny),
                        arrowprops=dict(arrowstyle="->",
                                       color=VEC, lw=0.8), zorder=6)
            lbl = f"{ac['callsign'][:6]}\n{int(ac['altitude'])//1000}k"
            ax.text(nx+0.022, ny+0.022, lbl, color=LBL,
                    fontsize=2.8, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc=BG, ec="none", alpha=0.5))

        plt.tight_layout(pad=0)
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight",
                    facecolor=BG, pad_inches=0)
        plt.close(fig)

        Image.open(save_path).convert("RGB").resize(
            (self.image_size, self.image_size),
            Image.LANCZOS).save(save_path)

    def _normalise(self, aircraft):
        lats = [a["latitude"]  for a in aircraft]
        lons = [a["longitude"] for a in aircraft]
        lr   = max(max(lons)-min(lons), 1e-3)
        latr = max(max(lats)-min(lats), 1e-3)
        out  = []
        for a in aircraft:
            nx = 0.07 + (a["longitude"]-min(lons))/lr   * 0.86
            ny = 0.07 + (a["latitude"] -min(lats))/latr * 0.86
            out.append((max(0.07,min(0.93,nx)),
                        max(0.07,min(0.93,ny))))
        return out

    def _layout_fp(self, aircraft):
        key = "|".join(sorted(
            f"{a['latitude']:.2f},{a['longitude']:.2f}"
            for a in aircraft))
        return hashlib.md5(key.encode()).hexdigest()

    def get_norm_coords(self, aircraft):
        coords = self._normalise(aircraft)
        return [{**a, "nx": nx, "ny": ny}
                for a, (nx, ny) in zip(aircraft, coords)]
