"""Voice and Text Query Interfaces for CNN engine."""

import os, logging, tempfile
import numpy as np
from typing import Dict
from PIL import Image

log = logging.getLogger(__name__)


class TextQueryInterface:
    def __init__(self, engine):
        self.engine = engine

    def query(self, image_path: str, question: str) -> Dict:
        img    = Image.open(image_path).convert("RGB")
        result = self.engine.answer(img, question)
        result["transcript"] = question
        return result


class VoiceQueryInterface:
    def __init__(self, engine, samplerate=16000, channels=1):
        self.engine     = engine
        self.samplerate = samplerate
        self.channels   = channels

    def record(self, duration=4.0) -> np.ndarray:
        import sounddevice as sd
        audio = sd.rec(int(duration * self.samplerate),
                       samplerate=self.samplerate,
                       channels=self.channels, dtype="float32")
        sd.wait()
        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        import speech_recognition as sr
        import scipy.io.wavfile as wav
        recogniser = sr.Recognizer()
        with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False) as tmp:
            path = tmp.name
            wav.write(path, self.samplerate,
                      (audio * 32767).astype(np.int16))
        try:
            with sr.AudioFile(path) as src:
                data = recogniser.record(src)
            return recogniser.recognize_google(data)
        except Exception as e:
            log.warning(f"STT error: {e}")
            return ""
        finally:
            os.unlink(path)

    def query_once(self, image_path: str,
                   duration=4.0) -> Dict:
        audio  = self.record(duration)
        text   = self.transcribe(audio)
        if not text:
            return {"transcript": "",
                    "answer": "Could not understand audio.",
                    "route": "none"}
        img    = Image.open(image_path).convert("RGB")
        result = self.engine.answer(img, text)
        result["transcript"] = text
        return result
