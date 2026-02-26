from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


Face = Tuple[int, int, int, int]


@dataclass
class EmotionResult:
    label: str
    confidence: float
    raw_label: str


class EmotionClassifier:
    def __init__(self) -> None:
        try:
            from fer import FER  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Emotion model dependency missing. Install with: pip install fer tensorflow-cpu"
            ) from exc
        self.model = FER(mtcnn=False)

    @staticmethod
    def _normalize_label(raw_label: str) -> str:
        mapping = {
            "happy": "happy",
            "sad": "sad",
            "angry": "angry",
            "fear": "nervous",
            "surprise": "surprised",
            "neutral": "neutral",
            "disgust": "disgusted",
        }
        return mapping.get(raw_label.lower(), raw_label.lower())

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

        try:
            top = self.model.top_emotion(roi)
        except Exception:
            return None

        if not top:
            return None

        raw_label, confidence = top
        if raw_label is None:
            return None

        normalized = self._normalize_label(str(raw_label))
        return EmotionResult(
            label=normalized,
            confidence=float(confidence or 0.0),
            raw_label=str(raw_label),
        )
