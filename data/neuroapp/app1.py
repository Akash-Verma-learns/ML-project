"""
NeuroApp - Flask backend for Emotion + Brain Tumor inference.
Run:  python app.py
"""

import os, io, base64
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as tv_models

# =====================================================
# CONFIG
# =====================================================

UPLOAD_FOLDER  = Path("uploads")
OUTPUT_FOLDER  = Path("outputs")
CHECKPOINT_DIR = Path("checkpoints")

EMOTION_CKPT   = Path(r"C:\Users\HP\ML_project\emotion_model_v3.pth")
TUMOR_CKPT     = Path(r"C:\Users\HP\ML_project\tumor_cnn_best_new.pt")

ALLOWED_EXT    = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TUMOR_CLASSES   = ["meningioma", "glioma", "pituitary"]   # updated: 3 classes
EMOTION_EMOJI   = {
    "angry": "😡", "disgust": "🤢", "fear": "😨",
    "happy": "😄", "neutral": "😐", "sad": "😢", "surprise": "😲"
}

# Emotion model dims (unchanged)
FACE_DIM   = 1408
HIDDEN_DIM = 256
N_EMO      = len(EMOTION_CLASSES)
N_TUMOR    = len(TUMOR_CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[NeuroApp] Device: {DEVICE}")

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]        = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"]   = 16 * 1024 * 1024


def allowed_file(filename):
    if not filename:
        return False
    return Path(filename).suffix.lower() in ALLOWED_EXT


# =====================================================
# MODEL DEFINITIONS
# =====================================================

# ── Emotion model (unchanged from original) ─────────

class FaceModel(nn.Module):
    """EfficientNet-B2 backbone matching face_enc.* keys. Output: [B, 1408]"""
    def __init__(self):
        super().__init__()
        base          = tv_models.efficientnet_b2(weights=None)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


# ── Tumor model (updated: EfficientNet-B0, 3 classes) ──

class TumorMRIEncoder(nn.Module):
    """EfficientNet-B0 backbone. Output: [B, 1280]"""
    def __init__(self):
        super().__init__()
        base          = tv_models.efficientnet_b0(weights=None)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.drop     = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.drop(x.view(x.size(0), -1))   # (B, 1280)


class FMRIEncoder(nn.Module):
    """Matches fmri_enc in tumor_cnn_train.py. Input: 128-d -> Output: 64-d"""
    def __init__(self, in_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BrainTumorCNN(nn.Module):
    """
    Matches tumor_cnn_train.py exactly including fMRI branch:
      MRI  -> EfficientNet-B0 -> 1280-d
      fMRI -> FMRIEncoder     ->   64-d
      Concat -> FC(512) -> FC(256) -> FC(3)
    At inference time we pass zeros for fMRI (no subject data available).
    """
    def __init__(self, num_classes=3, fmri_in_dim=128):
        super().__init__()
        self.mri_enc  = TumorMRIEncoder()
        self.fmri_enc = FMRIEncoder(in_dim=fmri_in_dim)
        fused         = 1280 + 64   # 1344
        self.classifier = nn.Sequential(
            nn.Linear(fused, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, fmri=None):
        mri_feat = self.mri_enc(x)                          # (B, 1280)
        if fmri is None:
            # No fMRI data at inference — pass zeros
            fmri = torch.zeros(x.size(0), 128, device=x.device)
        fmri_feat = self.fmri_enc(fmri)                     # (B, 64)
        fused     = torch.cat([mri_feat, fmri_feat], dim=1) # (B, 1344)
        return self.classifier(fused)


# =====================================================
# LOAD MODELS
# =====================================================

face_model  = None
emotion_clf = None
tumor_model = None


def load_emotion_model():
    global face_model, emotion_clf

    face_model = FaceModel().to(DEVICE)

    if EMOTION_CKPT.exists():
        ckpt = torch.load(EMOTION_CKPT, map_location=DEVICE, weights_only=False)

        # Strip "face_enc." prefix to load into FaceModel
        face_state = {
            k.replace("face_enc.", ""): v
            for k, v in ckpt.items()
            if k.startswith("face_enc.")
        }
        missing, _ = face_model.load_state_dict(face_state, strict=False)
        if missing:
            print(f"[emotion] WARNING: {len(missing)} missing keys in face encoder")
        else:
            print(f"[emotion] Face encoder loaded cleanly")

        # Slice face-only columns from fused classifier weight [256, 1504]
        clf0_w = ckpt["classifier.0.weight"]   # [256, 1504]
        clf0_b = ckpt["classifier.0.bias"]     # [256]

        # Find the N_EMO-class output layer
        final_key = None
        for k, v in ckpt.items():
            if k.startswith("classifier.") and k.endswith(".weight") and v.shape[0] == N_EMO:
                final_key = k
                break

        if final_key is None:
            raise RuntimeError(
                f"[emotion] Cannot find {N_EMO}-class output layer in checkpoint."
            )

        final_w = ckpt[final_key]
        final_b = ckpt[final_key.replace(".weight", ".bias")]

        emotion_clf = nn.Sequential(
            nn.Linear(FACE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, N_EMO),
        ).to(DEVICE)

        emotion_clf[0].weight = nn.Parameter(clf0_w[:, :FACE_DIM].clone())
        emotion_clf[0].bias   = nn.Parameter(clf0_b.clone())
        emotion_clf[2].weight = nn.Parameter(final_w.clone())
        emotion_clf[2].bias   = nn.Parameter(final_b.clone())

        print(f"[emotion] Classifier: {FACE_DIM}->{HIDDEN_DIM}->{N_EMO} "
              f"(sliced [:, :{FACE_DIM}] from [256, 1504])")
        print(f"[emotion] Loaded from {EMOTION_CKPT}")

    else:
        emotion_clf = nn.Sequential(
            nn.Linear(FACE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, N_EMO),
        ).to(DEVICE)
        print(f"[emotion] Checkpoint not found - using random weights")

    face_model.eval()
    emotion_clf.eval()


def load_tumor_model():
    global tumor_model
    tumor_model = BrainTumorCNN(num_classes=N_TUMOR, fmri_in_dim=128).to(DEVICE)

    if TUMOR_CKPT.exists():
        try:
            state = torch.load(TUMOR_CKPT, map_location=DEVICE, weights_only=False)
            tumor_model.load_state_dict(state, strict=True)
            print(f"[tumor] Loaded from {TUMOR_CKPT}")
        except RuntimeError as e:
            print(f"[tumor] Load error: {e}")
            print(f"[tumor] Falling back to random weights")
    else:
        print(f"[tumor] Checkpoint not found at {TUMOR_CKPT} - using random weights")

    tumor_model.eval()


load_emotion_model()
load_tumor_model()


# =====================================================
# TRANSFORMS
# =====================================================

# EfficientNet-B2 native resolution for emotion
emotion_tf = T.Compose([
    T.Resize((260, 260)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# EfficientNet-B0 native resolution for tumor (updated from 96x96)
tumor_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =====================================================
# INFERENCE HELPERS
# =====================================================

def predict_emotion(pil_img: Image.Image):
    tensor = emotion_tf(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats  = face_model(tensor)           # [1, 1408]
        logits = emotion_clf(feats)           # [1, 7]
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    idx = int(probs.argmax())
    return {
        "label":      EMOTION_CLASSES[idx],
        "emoji":      EMOTION_EMOJI[EMOTION_CLASSES[idx]],
        "confidence": float(probs[idx]),
        "all_probs":  {k: float(v) for k, v in zip(EMOTION_CLASSES, probs)},
    }


def predict_tumor(pil_img: Image.Image, emotion_vec=None):
    import cv2

    tensor = tumor_tf(pil_img).unsqueeze(0).to(DEVICE)

    gradients  = [None]
    activations = [None]

    def fwd_hook(m, i, o):
        activations[0] = o.detach().clone()

    def bwd_hook(m, gi, go):
        gradients[0] = go[0].detach().clone()

    # Hook on last conv block of EfficientNet-B0
    last_conv = tumor_model.mri_enc.features[-1]
    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_full_backward_hook(bwd_hook)

    try:
        tumor_model.zero_grad()
        with torch.enable_grad():
            tensor_grad = tensor.requires_grad_(True)
            fmri_zeros  = torch.zeros(1, 128, device=DEVICE)
            logits = tumor_model(tensor_grad, fmri_zeros)
            probs  = torch.softmax(logits, dim=1).squeeze()
            idx    = int(probs.argmax().item())
            logits[0, idx].backward()
    finally:
        h1.remove()
        h2.remove()

    # Build Grad-CAM -- matches original 3-panel style
    grad    = gradients[0].squeeze(0)
    act     = activations[0].squeeze(0)
    weights = F.relu(grad).mean(dim=(1, 2))
    cam     = (weights[:, None, None] * act).sum(0)
    cam     = F.relu(cam)
    cam     = cam - cam.min()
    cam     = cam / (cam.max() + 1e-8)
    cam_np  = cam.cpu().numpy()

    orig_np     = np.array(pil_img.convert("RGB"))
    cam_resized = cv2.resize(cam_np, (orig_np.shape[1], orig_np.shape[0]),
                             interpolation=cv2.INTER_CUBIC)

    # Percentile threshold -- suppress bottom 30%
    threshold   = np.percentile(cam_resized, 70)
    cam_thresh  = np.where(cam_resized >= threshold, cam_resized, cam_resized * 0.2)
    cam_thresh  = (cam_thresh - cam_thresh.min()) / (cam_thresh.max() + 1e-8)
    cam_uint8   = (cam_thresh * 255).astype(np.uint8)

    # INFERNO colormap
    heat        = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_INFERNO)
    heat_rgb    = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    # Adaptive alpha blending
    alpha_map   = (0.35 + 0.35 * cam_thresh)[:, :, np.newaxis]
    overlay     = (orig_np * (1 - alpha_map) + heat_rgb * alpha_map).astype(np.uint8)
    heatmap_v   = (np.zeros_like(orig_np) * (1 - alpha_map) + heat_rgb * alpha_map).astype(np.uint8)

    # Contour panel
    contour_img = overlay.copy()
    mask        = (cam_thresh > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 180), 2)

    def to_b64(arr):
        buf = io.BytesIO()
        Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    probs_np = probs.cpu().detach().numpy()
    return {
        "label":       TUMOR_CLASSES[idx],
        "confidence":  float(probs_np[idx]),
        "all_probs":   {k: float(v) for k, v in zip(TUMOR_CLASSES, probs_np)},
        "gradcam_b64": to_b64(overlay),
        "heatmap_b64": to_b64(heatmap_v),
        "contour_b64": to_b64(contour_img),
    }


# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/outputs/<path:fname>")
def serve_output(fname):
    return send_from_directory(OUTPUT_FOLDER, fname)


@app.route("/api/emotion", methods=["POST"])
def api_emotion():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported type: '{Path(f.filename).suffix}'. "
                                 f"Allowed: {', '.join(sorted(ALLOWED_EXT))}"}), 400
    try:
        return jsonify(predict_emotion(Image.open(f.stream).convert("RGB")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tumor", methods=["POST"])
def api_tumor():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported type: '{Path(f.filename).suffix}'. "
                                 f"Allowed: {', '.join(sorted(ALLOWED_EXT))}"}), 400
    try:
        img            = Image.open(f.stream).convert("RGB")
        emotion_result = None
        try:
            emotion_result = predict_emotion(img)
        except:
            pass
        return jsonify({
            "tumor":   predict_tumor(img),
            "emotion": emotion_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/full", methods=["POST"])
def api_full():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported type: '{Path(f.filename).suffix}'. "
                                 f"Allowed: {', '.join(sorted(ALLOWED_EXT))}"}), 400
    try:
        img = Image.open(f.stream).convert("RGB")
        emo = predict_emotion(img)
        tum = predict_tumor(img, list(emo["all_probs"].values()))
        return jsonify({"emotion": emo, "tumor": tum})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"\n  NeuroApp running on http://127.0.0.1:5000  (device={DEVICE})\n")
    app.run(debug=True, port=5000)