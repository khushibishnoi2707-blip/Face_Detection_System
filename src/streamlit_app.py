from __future__ import annotations

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

try:
    from face_detector import DetectionConfig, FaceDetector
    from emotion_classifier import EmotionClassifier
except ModuleNotFoundError:
    from src.face_detector import DetectionConfig, FaceDetector
    from src.emotion_classifier import EmotionClassifier


class FaceVideoProcessor(VideoProcessorBase):
    def __init__(self, detector: FaceDetector, classifier: EmotionClassifier | None) -> None:
        self.detector = detector
        self.classifier = classifier

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        faces = self.detector.detect(image)
        labels: list[str] = []
        if self.classifier is not None:
            for face in faces:
                emotion = self.classifier.predict_on_face(image, face)
                if emotion is None:
                    labels.append("emotion: unknown")
                else:
                    labels.append(f"{emotion.label} ({emotion.confidence * 100:.0f}%)")
        annotated = self.detector.annotate(image, faces, labels=labels if labels else None)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def _detect_from_uploaded_bytes(
    detector: FaceDetector,
    classifier: EmotionClassifier | None,
    raw_bytes: bytes,
) -> np.ndarray:
    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image data from camera input.")
    faces = detector.detect(frame)
    labels: list[str] = []
    if classifier is not None:
        for face in faces:
            emotion = classifier.predict_on_face(frame, face)
            if emotion is None:
                labels.append("emotion: unknown")
            else:
                labels.append(f"{emotion.label} ({emotion.confidence * 100:.0f}%)")
    return detector.annotate(frame, faces, labels=labels if labels else None)


@st.cache_resource
def _load_emotion_classifier() -> EmotionClassifier:
    return EmotionClassifier()


def main() -> None:
    st.set_page_config(page_title="Live Face Detection", layout="centered")
    st.title("Live Face Detection")
    st.write("Start the camera stream and detect faces in real time.")

    st.sidebar.header("Detection Settings")
    scale_factor = st.sidebar.slider("Scale Factor", 1.01, 1.50, 1.10, 0.01)
    min_neighbors = st.sidebar.slider("Min Neighbors", 1, 15, 5, 1)

    config = DetectionConfig(
        scale_factor=float(scale_factor),
        min_neighbors=int(min_neighbors),
    )
    detector = FaceDetector(config=config)
    classifier: EmotionClassifier | None = None
    try:
        classifier = _load_emotion_classifier()
    except Exception as exc:
        st.warning(f"Emotion classification unavailable: {exc}")

    live_tab, fallback_tab = st.tabs(["Live (WebRTC)", "Fallback (Snapshot)"])

    with live_tab:
        st.info("Grant camera permission and click Start. If this tab stays black, use Fallback.")
        webrtc_streamer(
            key="face-detection-live",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "facingMode": "user"},
                "audio": False,
            },
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
            },
            video_processor_factory=lambda: FaceVideoProcessor(detector, classifier),
            async_processing=True,
        )

    with fallback_tab:
        st.write("Use this if live stream is blocked by browser/network policy.")
        shot = st.camera_input("Take a photo")
        if shot is not None:
            try:
                annotated = _detect_from_uploaded_bytes(detector, classifier, shot.getvalue())
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="Detected faces",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Failed to process snapshot: {exc}")

    st.caption("Face boxes are drawn using OpenCV Haar cascade detection.")


if __name__ == "__main__":
    main()
