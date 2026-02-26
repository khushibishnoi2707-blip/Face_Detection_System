# Face Detection System (OpenCV)

A practical face detection project using OpenCV's Haar Cascade classifier.

## Features
- Detect faces from webcam in real time.
- Detect faces in a single image and optionally save annotated output.
- Process full video files and optionally save annotated video output.
- Adjustable detector parameters for tuning sensitivity.

## Project Structure
```text
face_Detecrtion/
├─ src/
│  ├─ face_detector.py
│  └─ main.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Setup
1. Create and activate a virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage

### 1) Webcam mode
```powershell
python src/main.py --mode webcam
```
Press `q` to quit.

### 2) Image mode
```powershell
python src/main.py --mode image --input path\to\image.jpg --output output\annotated.jpg
```

### 3) Video mode
```powershell
python src/main.py --mode video --input path\to\video.mp4 --output output\annotated.mp4 --display
```

## Detector Tuning
You can tune detection with:
- `--scale-factor` (default `1.1`)
- `--min-neighbors` (default `5`)

Example:
```powershell
python src/main.py --mode webcam --scale-factor 1.2 --min-neighbors 6
```

## Git-Ready Commit
After verifying locally:
```powershell
git add .
git commit -m "Add OpenCV-based face detection system with webcam/image/video support"
```
