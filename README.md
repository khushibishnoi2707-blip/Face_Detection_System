# Face Detection System (OpenCV + Streamlit)

A practical face detection project using OpenCV's Haar Cascade classifier.

## Features
- Detect faces from webcam in real time (CLI).
- Detect faces in a single image and optionally save annotated output.
- Process full video files and optionally save annotated video output.
- Run live face detection in browser using Streamlit.
- Classify face emotion categories (happy, sad, angry, nervous, etc.).
- Adjustable detector parameters for tuning sensitivity.

## Project Structure
```text
face_Detecrtion/
|-- app.py
|-- src/
|   |-- face_detector.py
|   |-- main.py
|   `-- streamlit_app.py
|-- requirements.txt
|-- .gitignore
`-- README.md
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

## CLI Usage

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

To run without emotion labels:
```powershell
python src/main.py --mode webcam --disable-emotion
```

## Streamlit Live App
Run the web app:
```powershell
streamlit run app.py
```

Then open the local URL shown in terminal, allow camera access, and click `Start`.
If live video is black or stuck, open the `Fallback (Snapshot)` tab and use `Take a photo`.

For Streamlit Cloud deployment use:
- Branch: `main`
- Main file path: `app.py`

Note:
- Cloud deploy uses `opencv-python-headless` and `runtime.txt` (`python-3.11`) for compatibility.
- Emotion classification uses pinned `fer` + `tensorflow-cpu` + `keras` versions for cloud stability.

## Detector Tuning
You can tune detection with:
- `--scale-factor` (default `1.1`)
- `--min-neighbors` (default `5`)

CLI example:
```powershell
python src/main.py --mode webcam --scale-factor 1.2 --min-neighbors 6
```
