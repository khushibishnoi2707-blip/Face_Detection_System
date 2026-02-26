from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


Face = Tuple[int, int, int, int]


@dataclass
class DetectionConfig:
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: Tuple[int, int] = (30, 30)


class FaceDetector:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load cascade classifier from: {cascade_path}")

    def detect(self, frame: np.ndarray) -> List[Face]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=self.config.min_size,
        )
        return list(faces)

    def annotate(
        self,
        frame: np.ndarray,
        faces: List[Face],
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        annotated = frame.copy()
        for idx, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if labels and idx < len(labels) and labels[idx]:
                text = labels[idx]
                cv2.rectangle(
                    annotated,
                    (x, max(0, y - 25)),
                    (x + max(80, len(text) * 9), y),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    annotated,
                    text,
                    (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
        cv2.putText(
            annotated,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated
