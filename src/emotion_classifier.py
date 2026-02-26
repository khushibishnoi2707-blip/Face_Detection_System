from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


Face = Tuple[int, int, int, int]


@dataclass
class EmotionResult:
    label: str
    confidence: float
    raw_label: str


class EmotionClassifier:
    """
    Lightweight heuristic emotion classifier for cloud compatibility.
    Categories: happy, sad, angry, nervous, neutral.
    """

    def __init__(self) -> None:
        smile_path = cv2.data.haarcascades + "haarcascade_smile.xml"
        self.smile_cascade = cv2.CascadeClassifier(smile_path)
        if self.smile_cascade.empty():
            raise RuntimeError(f"Failed to load smile cascade: {smile_path}")

    def predict_on_face(self, frame: np.ndarray, face: Face) -> Optional[EmotionResult]:
        x, y, w, h = face
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        smiles = self.smile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.7,
            minNeighbors=20,
            minSize=(max(20, w // 6), max(20, h // 6)),
        )

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # Heuristic mapping
        if len(smiles) > 0:
            return EmotionResult(label="happy", confidence=0.85, raw_label="smile")
        if contrast > 58:
            return EmotionResult(label="angry", confidence=0.62, raw_label="high_contrast")
        if brightness < 85:
            return EmotionResult(label="sad", confidence=0.60, raw_label="low_brightness")
        if contrast > 45:
            return EmotionResult(label="nervous", confidence=0.58, raw_label="mid_high_contrast")
        return EmotionResult(label="neutral", confidence=0.55, raw_label="baseline")
