"""
OpenSky Network API Collector
- Loads existing frames from Drive
- Adds 2000 NEW unique frames (no duplicates with existing)
- Total target: 4000 unique frames
- Auto falls back to SyntheticOpenSkyCollector if API unavailable
"""

import os, json, time, hashlib, logging, requests
import numpy as np
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OPENSKY_URL = "https://opensky-network.org/api/states/all"

REGIONS = [
    {"lamin": 36.0, "lomin": -10.0, "lamax": 60.0, "lomax":  25.0},
    {"lamin": 25.0, "lomin":-125.0, "lamax": 55.0, "lomax": -65.0},
    {"lamin": 10.0, "lomin":  60.0, "lamax": 50.0, "lomax": 140.0},
    {"lamin": 15.0, "lomin":  30.0, "lamax": 40.0, "lomax":  65.0},
    {"lamin":-45.0, "lomin": 110.0, "lamax":-10.0, "lomax": 160.0},
    {"lamin":-35.0, "lomin": -80.0, "lamax": 15.0, "lomax": -30.0},
]


def _fingerprint(aircraft: List[Dict]) -> str:
    key = ",".join(sorted(
        f"{a['latitude']:.2f},{a['longitude']:.2f}"
        for a in aircraft))
    return hashlib.md5(key.encode()).hexdigest()


class OpenSkyCollector:
    """
    Collects NEW unique frames not already in existing dataset.
    Pass existing_frames to avoid duplicates with previous collection.
    """

    def __init__(self, output_dir: str = "data/raw",
                 num_new_frames: int = 2000,
                 existing_frames: List[Dict] = None,
                 aircraft_per_frame: Tuple[int, int] = (4, 10),
                 sleep_between: float = 2.0,
                 timeout: int = 15):
        self.output_dir    = output_dir
        self.num_new       = num_new_frames
        self.min_ac, self.max_ac = aircraft_per_frame
        self.sleep         = sleep_between
        self.timeout       = timeout
        os.makedirs(output_dir, exist_ok=True)

        # Load fingerprints of existing frames to avoid duplicates
        self._seen = set()
        if existing_frames:
            for fr in existing_frames:
                self._seen.add(_fingerprint(fr["aircraft"]))
            log.info(f"Loaded {len(self._seen)} existing fingerprints.")

    def collect(self, start_idx: int = 0) -> List[Dict]:
        frames, idx, ridx, fails = [], start_idx, 0, 0
        log.info(f"Collecting {self.num_new} NEW unique frames …")

        while len(frames) < self.num_new:
            bbox = REGIONS[ridx % len(REGIONS)]
            ridx += 1
            try:
                states = self._fetch(bbox)
            except Exception as e:
                fails += 1
                log.warning(f"API error ({fails}): {e}")
                if fails > 15:
                    log.error("Too many failures. Stopping.")
                    break
                time.sleep(5)
                continue

            fails = 0
            if not states:
                time.sleep(self.sleep)
                continue

            aircraft = self._parse(states)
            if len(aircraft) < self.min_ac:
                time.sleep(self.sleep)
                continue

            np.random.shuffle(aircraft)
            aircraft = aircraft[:self.max_ac]
            fp = _fingerprint(aircraft)
            if fp in self._seen:
                time.sleep(self.sleep)
                continue
            self._seen.add(fp)

            frame = {"frame_id":  f"frame_{idx:04d}",
                     "timestamp": int(time.time()),
                     "aircraft":  aircraft}
            frames.append(frame)
            self._save(frame)
            idx += 1
            if idx % 100 == 0:
                log.info(f"  {len(frames)}/{self.num_new} new frames …")
            time.sleep(self.sleep)

        log.info(f"Collected {len(frames)} new unique frames.")
        return frames

    def _fetch(self, bbox):
        r = requests.get(OPENSKY_URL, params=bbox, timeout=self.timeout)
        if r.status_code == 429:
            log.warning("Rate limited — sleeping 60s")
            time.sleep(60)
            return []
        r.raise_for_status()
        return r.json().get("states") or []

    def _parse(self, states) -> List[Dict]:
        out = []
        for s in states:
            if s[5] is None or s[6] is None or s[8]:
                continue
            alt = ((s[7] or 0) if s[7] else (s[13] or 0)) * 3.28084
            out.append({
                "icao24":   s[0] or "unk",
                "callsign": (s[1] or "N/A").strip(),
                "latitude":  round(float(s[6]), 5),
                "longitude": round(float(s[5]), 5),
                "altitude":  round(alt, 0),
                "heading":   round(float(s[10]), 1) if s[10] is not None else 0.0,
                "velocity":  round(float(s[9])*1.94384, 1) if s[9] is not None else 0.0,
            })
        return out

    def _save(self, frame):
        with open(os.path.join(self.output_dir,
                               f"{frame['frame_id']}.json"), "w") as f:
            json.dump(frame, f)


class SyntheticOpenSkyCollector(OpenSkyCollector):
    """
    Generates NEW unique synthetic frames.
    Skips any fingerprint already in existing_frames.
    Uses different random seeds per run so new frames differ from old ones.
    """

    def collect(self, start_idx: int = 0) -> List[Dict]:
        log.info(f"[SYNTHETIC] Generating {self.num_new} NEW unique frames …")

        # Use a different seed from the original collection (which used 42)
        rng = np.random.default_rng(seed=99)

        centres = [
            (50.0, 10.0), (40.0, -100.0), (35.0, 100.0),
            (25.0, 45.0), (-25.0, 135.0), (-10.0, -55.0),
            (55.0, -3.0), (35.0, 139.0),  (19.0, -99.0),
            (-34.0, 151.0), (48.0, 16.0), (1.0, 103.0),
        ]

        frames, attempts = [], 0

        while len(frames) < self.num_new:
            attempts += 1
            if attempts > self.num_new * 15:
                log.warning("Could not generate enough unique frames.")
                break

            clat, clon = centres[attempts % len(centres)]
            clat += float(rng.uniform(-25, 25))
            clon += float(rng.uniform(-30, 30))
            clat = max(-80.0, min(80.0, clat))
            clon = max(-179.0, min(179.0, clon))

            n_ac = int(rng.integers(self.min_ac, self.max_ac + 1))
            aircraft = []
            for j in range(n_ac):
                lat = float(rng.uniform(clat - 6, clat + 6))
                lon = float(rng.uniform(clon - 9, clon + 9))
                lat = max(-85.0, min(85.0, lat))
                lon = max(-179.9, min(179.9, lon))
                aircraft.append({
                    "icao24":   f"n{start_idx+len(frames):05d}{j}",
                    "callsign": f"NEW{start_idx+len(frames):04d}{j}",
                    "latitude":  round(lat, 5),
                    "longitude": round(lon, 5),
                    "altitude":  round(float(rng.uniform(5000, 45000)), 0),
                    "heading":   round(float(rng.uniform(0, 360)), 1),
                    "velocity":  round(float(rng.uniform(150, 600)), 1),
                })

            fp = _fingerprint(aircraft)
            if fp in self._seen:
                continue
            self._seen.add(fp)

            frame = {
                "frame_id":  f"frame_{start_idx + len(frames):04d}",
                "timestamp": 1800000000 + len(frames),
                "aircraft":  aircraft,
            }
            frames.append(frame)
            self._save(frame)

            if len(frames) % 200 == 0:
                log.info(f"  [SYNTHETIC] {len(frames)}/{self.num_new} …")

        log.info(f"[SYNTHETIC] Done. {len(frames)} new unique frames.")
        return frames
