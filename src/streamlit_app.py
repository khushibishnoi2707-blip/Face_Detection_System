from __future__ import annotations

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

try:
    from face_detector import DetectionConfig, FaceDetector
except ModuleNotFoundError:
    from src.face_detector import DetectionConfig, FaceDetector


class FaceVideoProcessor(VideoProcessorBase):
    def __init__(self, detector: FaceDetector) -> None:
        self.detector = detector

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        faces = self.detector.detect(image)
        annotated = self.detector.annotate(image, faces)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


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

    st.info("Grant camera permission in your browser. Click Start to begin.")

    webrtc_streamer(
        key="face-detection-live",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: FaceVideoProcessor(detector),
        async_processing=True,
    )

    st.caption("Face boxes are drawn using OpenCV Haar cascade detection.")


if __name__ == "__main__":
    main()
