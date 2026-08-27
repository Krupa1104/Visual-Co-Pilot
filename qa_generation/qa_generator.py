"""
Radar QA Generator - 7 diverse QA pairs per frame.
14 question types: counting, positional, boolean,
comparative, threshold, heading, altitude, region.
"""

import os, json, random, logging
from typing import List, Dict

log = logging.getLogger(__name__)
random.seed(0)

ALT_THRESHOLDS = [10000, 15000, 20000, 25000, 30000, 35000, 40000]


class RadarQAGenerator:
    def __init__(self):
        self._gens = [
            self._q_count,
            self._q_pos_left,  self._q_pos_right,
            self._q_pos_top,   self._q_pos_bottom,
            self._q_pos_center,
            self._q_bool_gt,
            self._q_comp_highest, self._q_comp_lowest,
            self._q_threshold,
            self._q_heading,
            self._q_alt_highest, self._q_alt_lowest,
            self._q_region,
        ]

    def generate_all(self, frames, qa_per_image=7,
                     output_path="data/annotations/annotations.jsonl",
                     append=False):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        all_records = []
        for frame in frames:
            ac  = self._add_norm(frame["aircraft"])
            img = frame.get("image_path",
                f"data/processed/images/{frame['frame_id']}.png")
            pairs = self._gen_frame(ac, frame["frame_id"],
                                    img, qa_per_image)
            all_records.extend(pairs)

        mode = "a" if append else "w"
        with open(output_path, mode) as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")

        log.info(f"{'Appended' if append else 'Wrote'} "
                 f"{len(all_records)} QA pairs → {output_path}")
        return all_records

    def _gen_frame(self, aircraft, frame_id,
                   img_path, qa_per_image):
        gens = list(self._gens)
        random.shuffle(gens)
        pairs, seen = [], set()
        for gen in gens:
            if len(pairs) >= qa_per_image:
                break
            result = gen(aircraft)
            if result is None:
                continue
            q, a, qt = result
            if qt in seen:
                continue
            seen.add(qt)
            pairs.append({"image_id": frame_id,
                          "image_path": img_path,
                          "question": q,
                          "answer": str(a).lower().strip(),
                          "question_type": qt})
        while len(pairs) < qa_per_image:
            q, a, qt = self._q_count(aircraft)
            pairs.append({"image_id": frame_id,
                          "image_path": img_path,
                          "question": q,
                          "answer": str(a),
                          "question_type": qt})
        return pairs

    def _q_count(self, ac):
        return ("How many aircraft are visible on the radar?",
                str(len(ac)), "counting")

    def _q_pos_left(self, ac):
        has = any(a["nx"] < 0.40 for a in ac)
        return ("Is there any aircraft on the left side of the radar?",
                "yes" if has else "no", "positional_left")

    def _q_pos_right(self, ac):
        has = any(a["nx"] > 0.60 for a in ac)
        return ("Is there any aircraft on the right side of the radar?",
                "yes" if has else "no", "positional_right")

    def _q_pos_top(self, ac):
        has = any(a["ny"] > 0.60 for a in ac)
        return ("Is there any aircraft in the upper portion of the radar?",
                "yes" if has else "no", "positional_top")

    def _q_pos_bottom(self, ac):
        has = any(a["ny"] < 0.40 for a in ac)
        return ("Is there any aircraft in the lower portion of the radar?",
                "yes" if has else "no", "positional_bottom")

    def _q_pos_center(self, ac):
        has = any(0.30 < a["nx"] < 0.70
                  and 0.30 < a["ny"] < 0.70 for a in ac)
        return ("Is there any aircraft near the center of the radar?",
                "yes" if has else "no", "positional_center")

    def _q_bool_gt(self, ac):
        t = random.choice([3, 4, 5, 6])
        return (f"Are there more than {t} aircraft on the radar?",
                "yes" if len(ac) > t else "no", "boolean_count")

    def _q_comp_highest(self, ac):
        best = max(ac, key=lambda a: a["altitude"])
        cs   = (best["callsign"]
                if best["callsign"] not in ("N/A", "")
                else best["icao24"])
        return ("What is the callsign of the aircraft with the highest altitude?",
                cs.upper(), "comparative_highest")

    def _q_comp_lowest(self, ac):
        best = min(ac, key=lambda a: a["altitude"])
        cs   = (best["callsign"]
                if best["callsign"] not in ("N/A", "")
                else best["icao24"])
        return ("What is the callsign of the aircraft with the lowest altitude?",
                cs.upper(), "comparative_lowest")

    def _q_threshold(self, ac):
        t     = random.choice(ALT_THRESHOLDS)
        count = sum(1 for a in ac if a["altitude"] > t)
        return (f"How many aircraft are flying above {t:,} feet?",
                str(count), "threshold")

    def _q_heading(self, ac):
        center = min(ac, key=lambda a: abs(a["nx"]-0.5)+abs(a["ny"]-0.5))
        hdg    = round(center["heading"] / 10) * 10
        return ("What is the approximate heading of the aircraft nearest to the center?",
                f"{int(hdg)} degrees", "heading")

    def _q_alt_highest(self, ac):
        best  = max(ac, key=lambda a: a["altitude"])
        alt_k = int(best["altitude"]) // 1000
        return ("What is the approximate altitude of the highest aircraft in thousands of feet?",
                f"{alt_k}k feet", "altitude_value")

    def _q_alt_lowest(self, ac):
        best  = min(ac, key=lambda a: a["altitude"])
        alt_k = int(best["altitude"]) // 1000
        return ("What is the approximate altitude of the lowest aircraft in thousands of feet?",
                f"{alt_k}k feet", "altitude_lowest")

    def _q_region(self, ac):
        q = {"top-left":0,"top-right":0,
             "bottom-left":0,"bottom-right":0}
        for a in ac:
            side = "left"   if a["nx"] < 0.5 else "right"
            vert = "top"    if a["ny"] > 0.5 else "bottom"
            q[f"{vert}-{side}"] += 1
        densest = max(q, key=q.get)
        return ("Which quadrant of the radar has the most aircraft?",
                densest, "region_density")

    def _add_norm(self, aircraft):
        if not aircraft or "nx" in aircraft[0]:
            return aircraft
        lats = [a["latitude"]  for a in aircraft]
        lons = [a["longitude"] for a in aircraft]
        lr   = max(max(lons)-min(lons), 1e-3)
        latr = max(max(lats)-min(lats), 1e-3)
        out  = []
        for a in aircraft:
            nx = 0.07 + (a["longitude"]-min(lons))/lr   * 0.86
            ny = 0.07 + (a["latitude"] -min(lats))/latr * 0.86
            out.append({**a,
                        "nx": max(0.07,min(0.93,nx)),
                        "ny": max(0.07,min(0.93,ny))})
        return out
