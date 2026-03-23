"""
NeuroApp - Flask backend with Transformer Fusion
=================================================
CHANGE FROM ORIGINAL:
  OLD: BrainTumorCNN  — torch.cat → Linear layers
  NEW: BrainTumorTransformer — Multi-Head Cross-Attention fusion

The CNN backbones (EfficientNet-B0/B2) are identical.
Only the fusion block changed, so Grad-CAM still works the same way.

Load the new checkpoint:
  TUMOR_CKPT = Path("tumor_transformer_best.pt")
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

UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")

EMOTION_CKPT  = Path(r"C:\Users\HP\ML_project\emotion_model_v3.pth")
TUMOR_CKPT    = Path(r"C:\Users\HP\ML_project\tumor_transformer_best.pt")  # ← new checkpoint

ALLOWED_EXT   = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TUMOR_CLASSES   = ["meningioma", "glioma", "pituitary"]
EMOTION_EMOJI   = {
    "angry": "😡", "disgust": "🤢", "fear": "😨",
    "happy": "😄", "neutral": "😐", "sad": "😢", "surprise": "😲"
}

FACE_DIM   = 1408
HIDDEN_DIM = 256
N_EMO      = len(EMOTION_CLASSES)
N_TUMOR    = len(TUMOR_CLASSES)

# Transformer fusion hyperparams — must match what you trained with
FUSION_DIM  = 256
NUM_HEADS   = 8
NUM_LAYERS  = 3
FFN_DIM     = 512
ATTN_DROP   = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[NeuroApp] Device: {DEVICE}")

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def allowed_file(filename):
    if not filename:
        return False
    return Path(filename).suffix.lower() in ALLOWED_EXT


# =====================================================
# EMOTION MODEL  (unchanged from original)
# =====================================================

class FaceModel(nn.Module):
    """EfficientNet-B2 backbone. Output: [B, 1408]"""
    def __init__(self):
        super().__init__()
        base          = tv_models.efficientnet_b2(weights=None)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


# =====================================================
# TUMOR MODEL — CNN Encoders (unchanged)
# =====================================================

class TumorMRIEncoder(nn.Module):
    """EfficientNet-B0 CNN backbone. Output: [B, 1280]  (unchanged)"""
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
    """128-d fMRI → 64-d  (unchanged)"""
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


# =====================================================
# ★ NEW: TRANSFORMER FUSION  (replaces torch.cat)
# =====================================================

class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention + self-attention + FFN layer.
    MRI token queries fMRI token — interaction is per-sample.
    """
    def __init__(self, dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)

        self.self_attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.norm_self  = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, mri_tok, fmri_tok):
        # Cross-attention: MRI queries fMRI
        mri_norm = self.norm_cross(mri_tok)
        attended, attn_w = self.cross_attn(
            query=mri_norm, key=fmri_tok, value=fmri_tok)
        mri_tok = mri_tok + attended

        # Self-attention
        mri_norm = self.norm_self(mri_tok)
        refined, _ = self.self_attn(mri_norm, mri_norm, mri_norm)
        mri_tok  = mri_tok + refined

        # FFN
        mri_norm = self.norm_ffn(mri_tok)
        mri_tok  = mri_tok + self.ffn(mri_norm)

        return mri_tok, attn_w


class TransformerFusion(nn.Module):
    """
    Multi-layer cross-attention fusion.
    Replaces: torch.cat([mri, fmri]) → Linear(1344→512) → Linear → output
    With:     CrossAttentionLayer × N → LayerNorm → Linear → output
    """
    def __init__(self, mri_dim=1280, fmri_dim=64,
                 fusion_dim=256, num_heads=8,
                 num_layers=3, ffn_dim=512,
                 num_classes=3, dropout=0.1):
        super().__init__()
        self.mri_proj  = nn.Sequential(
            nn.Linear(mri_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.fmri_proj = nn.Sequential(
            nn.Linear(fmri_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.layers = nn.ModuleList([
            CrossAttentionLayer(fusion_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, mri_feat, fmri_feat):
        mri_tok  = self.mri_proj(mri_feat).unsqueeze(1)    # (B, 1, 256)
        fmri_tok = self.fmri_proj(fmri_feat).unsqueeze(1)  # (B, 1, 256)
        for layer in self.layers:
            mri_tok, _ = layer(mri_tok, fmri_tok)
        fused  = mri_tok.squeeze(1)
        return self.classifier(fused)


class BrainTumorTransformer(nn.Module):
    """
    Full model: EfficientNet-B0 (CNN) + FMRIEncoder + TransformerFusion.
    Same interface as old BrainTumorCNN — drop-in replacement for app.py.
    """
    def __init__(self, num_classes=3, fmri_in_dim=128):
        super().__init__()
        self.mri_enc  = TumorMRIEncoder()
        self.fmri_enc = FMRIEncoder(in_dim=fmri_in_dim)
        self.fusion   = TransformerFusion(
            mri_dim     = 1280,
            fmri_dim    = 64,
            fusion_dim  = FUSION_DIM,
            num_heads   = NUM_HEADS,
            num_layers  = NUM_LAYERS,
            ffn_dim     = FFN_DIM,
            num_classes = num_classes,
            dropout     = ATTN_DROP,
        )

    def forward(self, x, fmri=None):
        mri_feat = self.mri_enc(x)
        if fmri is None:
            fmri = torch.zeros(x.size(0), 128, device=x.device)
        fmri_feat = self.fmri_enc(fmri)
        return self.fusion(mri_feat, fmri_feat)


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
        face_state = {
            k.replace("face_enc.", ""): v
            for k, v in ckpt.items()
            if k.startswith("face_enc.")
        }
        missing, _ = face_model.load_state_dict(face_state, strict=False)
        if missing:
            print(f"[emotion] WARNING: {len(missing)} missing keys")
        else:
            print("[emotion] Face encoder loaded cleanly")

        clf0_w = ckpt["classifier.0.weight"]
        clf0_b = ckpt["classifier.0.bias"]
        final_key = None
        for k, v in ckpt.items():
            if k.startswith("classifier.") and k.endswith(".weight") \
                    and v.shape[0] == N_EMO:
                final_key = k
                break

        if final_key is None:
            raise RuntimeError(
                f"[emotion] Cannot find {N_EMO}-class output layer in checkpoint.")

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
        print(f"[emotion] Loaded from {EMOTION_CKPT}")
    else:
        emotion_clf = nn.Sequential(
            nn.Linear(FACE_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, N_EMO),
        ).to(DEVICE)
        print("[emotion] Checkpoint not found — using random weights")

    face_model.eval()
    emotion_clf.eval()


def load_tumor_model():
    global tumor_model
    tumor_model = BrainTumorTransformer(
        num_classes=N_TUMOR, fmri_in_dim=128).to(DEVICE)

    if TUMOR_CKPT.exists():
        try:
            state = torch.load(TUMOR_CKPT, map_location=DEVICE, weights_only=False)
            tumor_model.load_state_dict(state, strict=True)
            print(f"[tumor] Transformer model loaded from {TUMOR_CKPT}")
        except RuntimeError as e:
            print(f"[tumor] Load error: {e}")
            print("[tumor] Falling back to random weights")
    else:
        print(f"[tumor] Checkpoint not found at {TUMOR_CKPT} — using random weights")

    tumor_model.eval()


load_emotion_model()
load_tumor_model()


# =====================================================
# TRANSFORMS  (unchanged)
# =====================================================

emotion_tf = T.Compose([
    T.Resize((260, 260)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tumor_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =====================================================
# INFERENCE HELPERS  (unchanged)
# =====================================================

def predict_emotion(pil_img: Image.Image):
    tensor = emotion_tf(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats  = face_model(tensor)
        logits = emotion_clf(feats)
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
    gradients   = [None]
    activations = [None]

    def fwd_hook(m, i, o):
        activations[0] = o.detach().clone()

    def bwd_hook(m, gi, go):
        gradients[0] = go[0].detach().clone()

    # Hook on CNN backbone's last conv block (unchanged — CNN is still there)
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

    # Grad-CAM (unchanged)
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
    threshold   = np.percentile(cam_resized, 70)
    cam_thresh  = np.where(cam_resized >= threshold,
                           cam_resized, cam_resized * 0.2)
    cam_thresh  = (cam_thresh - cam_thresh.min()) / (cam_thresh.max() + 1e-8)
    cam_uint8   = (cam_thresh * 255).astype(np.uint8)
    heat        = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_INFERNO)
    heat_rgb    = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    alpha_map   = (0.35 + 0.35 * cam_thresh)[:, :, np.newaxis]
    overlay     = (orig_np * (1 - alpha_map) + heat_rgb * alpha_map).astype(np.uint8)
    heatmap_v   = (np.zeros_like(orig_np) * (1 - alpha_map)
                   + heat_rgb * alpha_map).astype(np.uint8)
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
# ROUTES  (unchanged)
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
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported type '{Path(f.filename).suffix}'"}), 400
    try:
        return jsonify(predict_emotion(Image.open(f.stream).convert("RGB")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tumor", methods=["POST"])
def api_tumor():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported type '{Path(f.filename).suffix}'"}), 400
    try:
        img            = Image.open(f.stream).convert("RGB")
        emotion_result = None
        try:
            emotion_result = predict_emotion(img)
        except Exception:
            pass
        return jsonify({"tumor": predict_tumor(img), "emotion": emotion_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/full", methods=["POST"])
def api_full():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported type '{Path(f.filename).suffix}'"}), 400
    try:
        img = Image.open(f.stream).convert("RGB")
        emo = predict_emotion(img)
        tum = predict_tumor(img, list(emo["all_probs"].values()))
        return jsonify({"emotion": emo, "tumor": tum})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"\n  NeuroApp (Transformer Fusion) on http://127.0.0.1:5000  "
          f"(device={DEVICE})\n")
    app.run(debug=True, port=5000)