"""
CNN Pilot Demo GUI
Interactive ipywidgets GUI for Colab.
Uses CNN ResNet50 to read image pixels directly.
"""

import os, io, json, random
from datetime import datetime
from typing import Dict, Optional
import numpy as np
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output


class CNNPilotDemo:
    PRESETS = [
        "How many aircraft are visible on the radar?",
        "Is there any aircraft on the left side of the radar?",
        "Is there any aircraft on the right side of the radar?",
        "Is there any aircraft in the upper portion of the radar?",
        "Is there any aircraft in the lower portion of the radar?",
        "Is there any aircraft near the center of the radar?",
        "Are there more than 4 aircraft on the radar?",
        "What is the callsign of the aircraft with the highest altitude?",
        "What is the callsign of the aircraft with the lowest altitude?",
        "Which quadrant of the radar has the most aircraft?",
        "How many aircraft are flying above 30,000 feet?",
        "What is the approximate altitude of the highest aircraft in thousands of feet?",
    ]

    def __init__(self, engine, image_dir: str,
                 save_dir: str = "demo_sessions"):
        self.engine        = engine
        self.image_dir     = image_dir
        self.save_dir      = save_dir
        self.current_image: Optional[Image.Image] = None
        self.current_path:  Optional[str]         = None
        self.session_log:   list                  = []
        os.makedirs(save_dir, exist_ok=True)

    def show(self):
        # Header
        header = widgets.HTML("""
        <div style="background:linear-gradient(135deg,#0a1628,#1a3a5c);
                    padding:16px 20px;border-radius:10px;margin-bottom:10px">
          <h2 style="color:#00d4ff;margin:0;font-family:monospace;
                     letter-spacing:2px">
            ✈ CNN VISUAL CO-PILOT</h2>
          <p style="color:#8ab4d4;margin:4px 0 0 0;font-size:12px">
            ResNet50 reads raw image pixels → answer
            (no embeddings)</p>
        </div>""")

        self.img_out = widgets.Output(layout=widgets.Layout(
            width="300px", height="300px",
            border="2px solid #00d4ff",
            border_radius="8px"))

        self.pixel_info = widgets.HTML(
            "<span style='color:#8ab4d4;font-size:11px'>"
            "No image loaded</span>")

        btn_random = widgets.Button(
            description="🎲 Random Image",
            button_style="info",
            layout=widgets.Layout(width="150px"))
        btn_upload = widgets.Button(
            description="📁 Upload Image",
            button_style="warning",
            layout=widgets.Layout(width="150px"))
        self.upload_w = widgets.FileUpload(
            accept=".png,.jpg,.jpeg", multiple=False,
            layout=widgets.Layout(display="none"))

        self.q_input = widgets.Text(
            placeholder="Type question and press Enter …",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "0px"})

        btn_ask = widgets.Button(
            description="🛩 Ask CNN",
            button_style="success",
            layout=widgets.Layout(width="110px", height="36px"))

        self.answer_out = widgets.Output(
            layout=widgets.Layout(
                min_height="90px",
                border="1px solid #1a3a5c",
                border_radius="8px",
                padding="8px", margin="6px 0"))

        self.history_out = widgets.Output(
            layout=widgets.Layout(
                height="200px", overflow_y="auto",
                border="1px solid #1a3a5c",
                border_radius="8px", padding="8px"))

        btn_clear = widgets.Button(
            description="🗑 Clear",
            button_style="danger",
            layout=widgets.Layout(width="100px"))
        btn_export = widgets.Button(
            description="💾 Export",
            button_style="warning",
            layout=widgets.Layout(width="100px"))

        self.status = widgets.HTML(
            "<span style='color:#8ab4d4'>Ready ✓</span>")

        # Preset buttons
        preset_btns = []
        for q in self.PRESETS:
            short = q[:46] + "…" if len(q) > 46 else q
            b = widgets.Button(
                description=short, tooltip=q,
                layout=widgets.Layout(
                    width="auto", height="28px", margin="2px"))
            b.style.button_color = "#1a3a5c"
            b.on_click(lambda ev, question=q:
                       self._ask(question))
            preset_btns.append(b)

        rows = [widgets.HBox(preset_btns[i:i+2])
                for i in range(0, len(preset_btns), 2)]

        # Wire events
        btn_random.on_click(self._on_random)
        btn_upload.on_click(lambda _: display(self.upload_w))
        self.upload_w.observe(self._on_upload, names="value")
        btn_ask.on_click(lambda _: self._on_ask())
        self.q_input.on_submit(lambda _: self._on_ask())
        btn_clear.on_click(self._on_clear)
        btn_export.on_click(self._on_export)

        # Layout
        left = widgets.VBox([
            widgets.HTML(
                "<b style='color:#00d4ff'>Radar Image</b>"),
            self.img_out,
            widgets.HBox([btn_random, btn_upload]),
            self.upload_w, self.pixel_info,
        ], layout=widgets.Layout(
            width="320px", margin="0 16px 0 0"))

        right = widgets.VBox([
            widgets.HTML(
                "<b style='color:#00d4ff'>Ask CNN Co-Pilot</b>"),
            widgets.HBox([self.q_input, btn_ask]),
            widgets.HTML(
                "<p style='color:#8ab4d4;font-size:11px;"
                "margin:6px 0 3px'>Quick questions:</p>"),
            widgets.VBox(rows),
            widgets.HTML(
                "<b style='color:#00d4ff;margin-top:6px'>"
                "Answer</b>"),
            self.answer_out,
        ], layout=widgets.Layout(flex="1"))

        root = widgets.VBox([
            header,
            widgets.HBox([left, right],
                         layout=widgets.Layout(
                             align_items="flex-start")),
            widgets.HTML(
                "<hr style='border-color:#1a3a5c;margin:10px 0'>"),
            widgets.HTML(
                "<b style='color:#00d4ff'>Session History</b>"),
            self.history_out,
            widgets.HBox([btn_clear, btn_export]),
            self.status,
        ], layout=widgets.Layout(padding="14px"))

        display(root)
        self._on_random(None)

    # ── helpers ───────────────────────────────────────────────────

    def _load_image(self, path_or_bytes, label=""):
        try:
            if isinstance(path_or_bytes, str):
                img = Image.open(path_or_bytes).convert("RGB")
                self.current_path = path_or_bytes
            else:
                img = Image.open(
                    io.BytesIO(path_or_bytes)).convert("RGB")
                self.current_path = "uploaded"
            self.current_image = img
            arr = np.array(img.resize((224, 224)))
            with self.img_out:
                clear_output(wait=True)
                display(img.resize((280, 280)))
            self.pixel_info.value = (
                f"<span style='color:#8ab4d4;font-size:11px'>"
                f"📍 {label} | pixels: {arr.shape} | "
                f"min={arr.min()} max={arr.max()} "
                f"mean={arr.mean():.1f}</span>")
            self._set_status(f"Loaded: {label} ✓")
        except Exception as e:
            self._set_status(f"Error: {e}", "red")

    def _ask(self, question: str):
        if self.current_image is None:
            self._set_status("⚠️ Load an image first", "orange")
            return
        self._set_status("CNN reading pixels …")
        try:
            result = self.engine.answer(
                self.current_image, question)
            self._show_answer(question, result)
            self._add_history(question, result)
            self._set_status("Ready ✓")
        except Exception as e:
            self._set_status(f"Error: {e}", "red")

    def _show_answer(self, question: str, result: Dict):
        ans        = result["answer"]
        conf       = result["confidence"]
        conf_pct   = int(conf * 100)
        conf_color = ("#00b894" if conf_pct >= 70
                      else "#fdcb6e" if conf_pct >= 40
                      else "#d63031")
        html = f"""
        <div style="font-family:monospace;padding:6px">
          <div style="color:#8ab4d4;font-size:11px;
                      margin-bottom:4px">Q: {question}</div>
          <div style="display:flex;align-items:center;
                      gap:10px;margin-bottom:6px">
            <span style="background:#6c5ce7;color:white;
                         padding:2px 8px;border-radius:4px;
                         font-size:11px;font-weight:bold">
              CNN PIXEL READ</span>
            <span style="color:white;font-size:20px;
                         font-weight:bold">{ans}</span>
          </div>
          <div style="background:#1a3a5c;border-radius:4px;
                      height:8px;width:100%;margin-bottom:4px">
            <div style="background:{conf_color};
                        width:{conf_pct}%;height:8px;
                        border-radius:4px"></div>
          </div>
          <div style="color:#8ab4d4;font-size:10px">
            Confidence: {conf_pct}% &nbsp;|&nbsp;
            Source: ResNet50 CNN
          </div>
        </div>"""
        with self.answer_out:
            clear_output(wait=True)
            display(widgets.HTML(html))

    def _add_history(self, question: str, result: Dict):
        ts   = datetime.now().strftime("%H:%M:%S")
        ans  = result["answer"]
        conf = int(result["confidence"] * 100)
        self.session_log.append({
            "time": ts, "question": question,
            "answer": ans, "confidence": conf})
        row = f"""
        <div style="border-bottom:1px solid #1a3a5c;
                    padding:3px 0;font-family:monospace">
          <span style="color:#636e72;font-size:10px">{ts}</span>
          <span style="background:#6c5ce7;color:white;
                       padding:1px 5px;border-radius:3px;
                       font-size:10px;margin:0 5px">CNN</span>
          <span style="color:#8ab4d4;font-size:11px">
            {question[:45]}</span><br>
          <span style="color:white;font-size:12px;
                       padding-left:80px">→ {ans}
            <span style="color:#636e72"> ({conf}%)</span>
          </span>
        </div>"""
        with self.history_out:
            display(widgets.HTML(row))

    def _set_status(self, msg, color="#8ab4d4"):
        self.status.value = (
            f"<span style='color:{color};"
            f"font-size:12px'>{msg}</span>")

    # ── event handlers ────────────────────────────────────────────

    def _on_random(self, _):
        try:
            files = [f for f in os.listdir(self.image_dir)
                     if f.endswith((".png",".jpg",".jpeg"))]
            if not files:
                self._set_status("No images found", "orange")
                return
            fname = random.choice(files)
            self._load_image(
                os.path.join(self.image_dir, fname), fname)
        except Exception as e:
            self._set_status(f"Error: {e}", "red")

    def _on_upload(self, change):
        if not change["new"]: return
        name    = list(change["new"].keys())[0]
        content = change["new"][name]["content"]
        self._load_image(content, name)

    def _on_ask(self):
        q = self.q_input.value.strip()
        if q:
            self._ask(q)
            self.q_input.value = ""

    def _on_clear(self, _):
        self.session_log.clear()
        with self.history_out: clear_output()
        with self.answer_out:  clear_output()
        self._set_status("Cleared ✓")

    def _on_export(self, _):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.save_dir,
                            f"session_{ts}.json")
        with open(path, "w") as f:
            json.dump(self.session_log, f, indent=2)
        self._set_status(f"Exported → {path}")
