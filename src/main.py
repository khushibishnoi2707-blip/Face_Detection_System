from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2

from face_detector import DetectionConfig, FaceDetector
from emotion_classifier import EmotionClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face detection using OpenCV Haar cascades.")
    parser.add_argument(
        "--mode",
        choices=["webcam", "image", "video"],
        required=True,
        help="Input mode.",
    )
    parser.add_argument("--input", type=str, help="Path to image/video file.")
    parser.add_argument("--output", type=str, help="Path to save annotated image/video.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display output window while processing.",
    )
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Cascade scale factor.")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Cascade min neighbors.")
    parser.add_argument(
        "--disable-emotion",
        action="store_true",
        help="Disable emotion classification labels.",
    )
    return parser.parse_args()


def _emotion_labels(
    classifier: Optional[EmotionClassifier],
    frame,
    faces,
) -> List[str]:
    if classifier is None:
        return []
    labels: List[str] = []
    for face in faces:
        result = classifier.predict_on_face(frame, face)
        if result is None:
            labels.append("emotion: unknown")
            continue
        labels.append(f"{result.label} ({result.confidence * 100:.0f}%)")
    return labels


def process_image(
    detector: FaceDetector,
    classifier: Optional[EmotionClassifier],
    input_path: Path,
    output_path: Path | None,
) -> None:
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    faces = detector.detect(frame)
    labels = _emotion_labels(classifier, frame, faces)
    annotated = detector.annotate(frame, faces, labels=labels)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved annotated image to: {output_path}")
    print(f"Detected {len(faces)} face(s) in image.")

    cv2.imshow("Face Detection - Image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_webcam(
    detector: FaceDetector,
    classifier: Optional[EmotionClassifier],
    camera_index: int,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {camera_index}")

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detector.detect(frame)
        labels = _emotion_labels(classifier, frame, faces)
        annotated = detector.annotate(frame, faces, labels=labels)
        cv2.imshow("Face Detection - Webcam", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_video(
    detector: FaceDetector,
    classifier: Optional[EmotionClassifier],
    input_path: Path,
    output_path: Path | None,
    display: bool,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot read video: {input_path}")

    writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detector.detect(frame)
        labels = _emotion_labels(classifier, frame, faces)
        annotated = detector.annotate(frame, faces, labels=labels)

        if writer:
            writer.write(annotated)
        if display:
            cv2.imshow("Face Detection - Video", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_count += 1

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video to: {output_path}")
    if display:
        cv2.destroyAllWindows()
    print(f"Processed {frame_count} frame(s).")


def main() -> None:
    args = parse_args()
    config = DetectionConfig(
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
    )
    detector = FaceDetector(config=config)
    classifier: Optional[EmotionClassifier] = None
    if not args.disable_emotion:
        try:
            classifier = EmotionClassifier()
            print("Emotion classification enabled.")
        except Exception as exc:
            print(f"Emotion classification unavailable: {exc}")
            print("Continuing with face detection only.")

    if args.mode == "image":
        if not args.input:
            raise ValueError("--input is required in image mode.")
        process_image(
            detector,
            classifier,
            Path(args.input),
            Path(args.output) if args.output else None,
        )
        return

    if args.mode == "webcam":
        process_webcam(detector, classifier, args.camera_index)
        return

    if not args.input:
        raise ValueError("--input is required in video mode.")
    process_video(
        detector,
        classifier,
        Path(args.input),
        Path(args.output) if args.output else None,
        args.display,
    )


if __name__ == "__main__":
    main()
