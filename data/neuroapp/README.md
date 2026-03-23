# NeuroLens — Web App for Emotion & Brain Tumor Analysis

A Flask web app that wraps your ML pipeline with a clean UI.

## Project Structure

```
neuroapp/
├── app.py                  # Flask backend
├── requirements.txt
├── templates/
│   └── index.html          # Frontend UI
├── uploads/                # Temp uploads (auto-created)
└── outputs/                # Grad-CAM outputs (auto-created)
```

## Setup

### 1. Copy your model files next to app.py

```
neuroapp/
├── emotion_model.pth           ← from your ML_project root
├── checkpoints/
│   └── tumor_cnn_best.pt       ← from your ML_project checkpoints/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Features

| Tab | Input | Output |
|-----|-------|--------|
| **Emotion** | Face photo | 7-class emotion + confidence bars |
| **Tumor MRI** | Brain MRI image | 4-class tumor + Grad-CAM saliency |
| **Full Analysis** | Any image | Both emotion + tumor together |

## Notes

- If `emotion_model.pth` is missing, the model uses random weights (for demo).
- If `checkpoints/tumor_cnn_best.pt` is missing, same applies.
- The tumor model uses the emotion prediction as a conditioning vector (as in your `hybrid.py`).
- Not for clinical use — research prototype only.
