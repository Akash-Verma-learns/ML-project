import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

import pickle
import nibabel as nib
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# PATHS
# =====================================

DATA_DIR = r"C:\Users\HP\ML_project\data"

FER_PATH    = os.path.join(DATA_DIR, "FER2013")
DEAP_PATH   = os.path.join(DATA_DIR, "deap-dataset", "data_preprocessed_python")
FMRI_PATH   = os.path.join(DATA_DIR, "fmri")

MODEL_PATH  = "emotion_model_v3.pth"

# =====================================
# DEVICE
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================
# CONFIG  (tune here without touching code below)
# =====================================

CFG = dict(
    batch_size   = 16,      # 16 safe for 4GB VRAM with EfficientNet-B2
    epochs       = 10,
    lr           = 3e-4,
    weight_decay = 1e-4,
    dropout      = 0.4,
    grad_clip    = 1.0,
    eeg_dim      = 32 * 5,  # 32 channels × 5 frequency bands
    fmri_dim     = 32,
    face_dim     = 1408,    # EfficientNet-B2 output dim (vs 1280 for MobileNetV2)
    num_classes  = 7,
    img_size     = 260,     # EfficientNet-B2 native resolution
)

# =====================================
# FER2013 — with augmentation
# =====================================

# Training: augmentation at EfficientNet-B2 native resolution (260x260)
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Test: deterministic, same normalisation
test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

train_data   = ImageFolder(os.path.join(FER_PATH, "train"), transform=train_transform)
test_data    = ImageFolder(os.path.join(FER_PATH, "test"),  transform=test_transform)

# ---- Weighted sampler — fixes class imbalance in FER2013 ----
# Counts how many samples exist per class, then assigns each sample
# an inverse-frequency weight so rare classes (disgust, fear) are
# seen as often as common ones (happy, neutral) during training.
class_counts  = np.bincount([s[1] for s in train_data.samples])
class_weights = 1.0 / class_counts
sample_weights = torch.tensor(
    [class_weights[s[1]] for s in train_data.samples], dtype=torch.float
)
sampler = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(sample_weights),
    replacement = True,
)
print("Class counts:", dict(zip(train_data.classes, class_counts)))

train_loader = DataLoader(train_data, batch_size=CFG["batch_size"],
                          sampler=sampler,          # replaces shuffle=True
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=CFG["batch_size"],
                          num_workers=0, pin_memory=True)

emotion_labels = train_data.classes
print("Emotion classes:", emotion_labels)

# =====================================
# FACE ENCODER  (EfficientNet-B2)
# =====================================
# EfficientNet-B2 outputs 1408-d features vs MobileNetV2's 1280-d.
# Better at fine-grained differences (disgust vs angry, fear vs sad).
# Native resolution 260x260 — much richer spatial features than 48x48.

class FaceEncoder(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        base             = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.features    = base.features
        self.pool        = nn.AdaptiveAvgPool2d((1, 1))
        self.drop        = nn.Dropout(p=dropout)

    def forward(self, x):                       # x: (B, 1, 260, 260)
        x = x.repeat(1, 3, 1, 1)               # → (B, 3, 260, 260)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)              # (B, 1408)
        return self.drop(x)

# =====================================
# EEG FEATURE EXTRACTION — band power
# =====================================
# 5 bands × 32 channels = 160-d vector per trial (vs 32-d before)

BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

def bandpower(psd, freqs, low, high):
    idx = np.where((freqs >= low) & (freqs < high))[0]
    return np.mean(psd[idx]) if len(idx) else 0.0

def extract_eeg_features():
    features, labels = [], []

    for file in os.listdir(DEAP_PATH):
        path = os.path.join(DEAP_PATH, file)
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        eeg = data["data"][:, :32, :]          # (40 trials, 32 ch, 8064 samples)
        lab = data["labels"]

        for trial in range(eeg.shape[0]):
            psd_features = []
            for ch in range(32):
                freqs, psd = welch(eeg[trial, ch], fs=128, nperseg=256)
                for low, high in BANDS.values():
                    psd_features.append(bandpower(psd, freqs, low, high))
            features.append(psd_features)
            labels.append(lab[trial][0])        # valence as proxy label

    X = np.array(features, dtype=np.float32)
    y = np.array(labels,   dtype=np.float32)

    # Binarise valence → 0 (low) / 1 (high) for EEG auxiliary task
    y_bin = (y > 5).astype(np.int64)

    # Z-score across trials — critical for stable training
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y_bin

print("Extracting EEG features (band power)...")
eeg_X, eeg_y = extract_eeg_features()
print(f"  EEG feature matrix: {eeg_X.shape}")   # should be (N_trials, 160)

# =====================================
# fMRI FEATURE EXTRACTION — regional means
# =====================================
# FIX: extract per-volume means along time axis so each subject gives
# a richer feature vector rather than one scalar.

def extract_fmri_features(target_dim=32):
    features = []

    for sub in os.listdir(FMRI_PATH):
        sub_path = os.path.join(FMRI_PATH, sub)
        if not os.path.isdir(sub_path):
            continue
        for file in os.listdir(sub_path):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                img  = nib.load(os.path.join(sub_path, file))
                data = img.get_fdata()           # (X, Y, Z) or (X, Y, Z, T)
                # Flatten spatial dims, average if 4-D
                if data.ndim == 4:
                    vec = data.reshape(-1, data.shape[-1]).mean(axis=1)
                else:
                    vec = data.flatten()
                # Subsample / pad to fixed length
                if len(vec) >= target_dim:
                    vec = vec[:target_dim]
                else:
                    vec = np.pad(vec, (0, target_dim - len(vec)))
                features.append(vec.astype(np.float32))

    if not features:
        print("  No fMRI files found — fMRI branch will be disabled.")
        return None

    X = np.array(features)
    # Z-score
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X

print("Extracting fMRI features...")
fmri_X = extract_fmri_features(target_dim=CFG["fmri_dim"])
if fmri_X is not None:
    print(f"  fMRI feature matrix: {fmri_X.shape}")

# =====================================
# MULTIMODAL FUSION MODEL
# =====================================
# Late-fusion: face + EEG encoders → concatenate → MLP classifier
# fMRI is optional (many users won't have paired data).

class EEGEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)                      # (B, 64)


class FusionClassifier(nn.Module):
    """
    Fuses face (1280-d) + EEG (64-d) [+ optional fMRI (fmri_dim-d)]
    into a single emotion prediction.
    """
    def __init__(self, use_fmri=False, fmri_dim=32,
                 num_classes=7, dropout=0.4):
        super().__init__()
        self.face_enc = FaceEncoder(dropout=dropout)
        self.eeg_enc  = EEGEncoder(CFG["eeg_dim"], dropout=dropout)

        self.use_fmri = use_fmri
        fused_dim = 1408 + 64               # EfficientNet-B2 (1408) + EEG (64)
        if use_fmri:
            self.fmri_enc = nn.Sequential(
                nn.Linear(fmri_dim, 32),
                nn.ReLU(),
            )
            fused_dim += 32

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, eeg, fmri=None):
        face_feat = self.face_enc(img)          # (B, 1280)
        eeg_feat  = self.eeg_enc(eeg)           # (B, 64)
        parts     = [face_feat, eeg_feat]

        if self.use_fmri and fmri is not None:
            parts.append(self.fmri_enc(fmri))   # (B, 32)

        fused  = torch.cat(parts, dim=1)
        logits = self.classifier(fused)
        return logits


use_fmri = fmri_X is not None
model    = FusionClassifier(
    use_fmri    = use_fmri,
    fmri_dim    = CFG["fmri_dim"],
    num_classes = CFG["num_classes"],
    dropout     = CFG["dropout"],
).to(device)

print(f"\nModel ready  |  use_fmri={use_fmri}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# =====================================
# TRAINING (face-only path for FER2013)
# NOTE: Full fusion requires paired face+EEG samples.
#       FER2013 doesn't have paired EEG, so we train face path on FER2013
#       and EEG encoder separately on DEAP, then fine-tune together.
# =====================================

# =====================================
# FOCAL LOSS
# =====================================
# Focal loss down-weights easy/confident predictions (happy, surprise)
# and focuses training on hard confused ones (disgust, fear, angry, sad).
# gamma=2 is standard; higher = more focus on hard examples.

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # per-class weights

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt      = torch.exp(-ce_loss)                    # confidence of correct class
        focal   = (1 - pt) ** self.gamma * ce_loss      # penalise easy examples less
        return focal.mean()

# ---- PHASE 1: train face encoder on FER2013 ----

# ---- Load old face weights if checkpoint exists, train everything else fresh ----

if os.path.exists(MODEL_PATH):
    print("\nFound existing checkpoint — loading old face encoder weights only...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    if "face" in checkpoint:
        # Old format: {"face": ..., "classifier": ...}
        # Remap old face keys → new face_enc sub-module structure
        face_state = checkpoint["face"]
        remapped = {}
        for k, v in face_state.items():
            new_key = k if k.startswith("features.") else "features." + k
            # Old model had Conv2d(1→32) for grayscale; new model expects Conv2d(3→32).
            # Expand the weight by repeating the single input channel 3 times
            # and scaling down so the sum stays equivalent.
            if new_key == "features.0.0.weight" and v.shape[1] == 1:
                v = v.repeat(1, 3, 1, 1) / 3.0
            remapped[new_key] = v

        missing, unexpected = model.face_enc.load_state_dict(remapped, strict=False)
        print(f"  Face encoder loaded  |  missing: {len(missing)}  unexpected: {len(unexpected)}")

        # Warm-start the fusion head bias from old classifier bias if shapes match
        if "classifier" in checkpoint:
            old_bias = checkpoint["classifier"]["bias"]   # (7,)
            new_final = model.classifier[-1]
            if new_final.bias.shape == old_bias.shape:
                with torch.no_grad():
                    new_final.bias.copy_(old_bias)
                print("  Old classifier bias transferred to fusion head.")

        print("  EEG encoder, fMRI encoder, fusion MLP → training from scratch.")

    else:
        # Already new flat format — load fully
        print("  Checkpoint is new format — loading full model...")
        model.load_state_dict(checkpoint, strict=False)

else:
    print("\nNo checkpoint found — training from scratch.")

# ---- Optimizer, Scheduler, Loss ----

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = CFG["lr"],
    weight_decay = CFG["weight_decay"],
)

scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-6)

# Per-class weights: tuned conservatively — too aggressive causes collapse.
# Order: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
manual_weights = torch.tensor([
    1.5,   # angry
    2.0,   # disgust  ← boosted but not overwhelming
    1.5,   # fear
    0.8,   # happy    ← slight reduction, not too harsh
    1.0,   # neutral
    1.5,   # sad
    0.8,   # surprise ← slight reduction
], device=device)

criterion = FocalLoss(weight=manual_weights, gamma=1.0)  # gamma 1.0 instead of 2.0 — gentler

print("\n--- Training fusion model on FER2013 ---")

for epoch in range(CFG["epochs"]):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for imgs, labels in train_loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        eeg_dummy  = torch.zeros(imgs.size(0), CFG["eeg_dim"],  device=device)
        fmri_dummy = torch.zeros(imgs.size(0), CFG["fmri_dim"], device=device) if use_fmri else None

        optimizer.zero_grad()
        logits = model(imgs, eeg_dummy, fmri_dummy)
        loss   = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        optimizer.step()

        total_loss += loss.item()
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    scheduler.step()

    train_acc  = 100 * correct / total
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1:02d}/{CFG['epochs']}  "
          f"Loss: {total_loss:.3f}  "
          f"Train Acc: {train_acc:.1f}%  "
          f"LR: {current_lr:.2e}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

# =====================================
# EVALUATION
# =====================================

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs       = imgs.to(device)
        eeg_dummy  = torch.zeros(imgs.size(0), CFG["eeg_dim"],  device=device)
        fmri_dummy = torch.zeros(imgs.size(0), CFG["fmri_dim"], device=device) if use_fmri else None
        logits     = model(imgs, eeg_dummy, fmri_dummy)
        preds      = logits.argmax(1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred,
                             target_names=emotion_labels, zero_division=0))

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title("Emotion Recognition — Confusion Matrix", fontsize=16)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix_v2.png", dpi=150)
plt.show()
print("Confusion matrix saved → confusion_matrix_v2.png")