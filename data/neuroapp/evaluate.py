"""
evaluate.py — Evaluate Emotion and Tumor models.
==================================================
Fixed to work with the new app.py architecture:
  - Emotion model: FaceModel + emotion_clf (not a single emotion_model)
  - Tumor model:   BrainTumorTransformer (not BrainTumorCNN)
  - No using_legacy_* flags — removed, not needed
  - evaluate() function defined here, not imported from app

Usage:
    cd C:\\Users\\HP\\ML_project\\data\\neuroapp
    python evaluate.py

Requirements:
    pip install scikit-learn seaborn matplotlib
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# ── Import models and config from app.py ─────────────────────────────────────
from app import (
    face_model,
    emotion_clf,
    tumor_model,
    EMOTION_CLASSES,
    TUMOR_CLASSES,
    emotion_tf,
    tumor_tf,
    DEVICE,
)

# ═════════════════════════════════════════════════════════════
#  PATHS — update these to match your local dataset locations
# ═════════════════════════════════════════════════════════════

EMOTION_VAL_PATH = r"C:\Users\HP\ML_project\data\fer2013\test"

# Uncomment once you have the tumor dataset:
# TUMOR_VAL_PATH = r"C:\Users\HP\ML_project\data\tumor\Testing"


# ═════════════════════════════════════════════════════════════
#  EVALUATE FUNCTION
# ═════════════════════════════════════════════════════════════

def evaluate(dataloader, class_names, is_tumor=False, save_path="confusion_matrix.png"):
    """
    Run evaluation on a DataLoader.

    For emotion: uses face_model + emotion_clf from app.py
    For tumor:   uses tumor_model (BrainTumorTransformer) from app.py

    Args:
        dataloader : torch DataLoader yielding (images, labels)
        class_names: list of class name strings
        is_tumor   : True to use tumor_model, False for emotion
        save_path  : filename for the confusion matrix PNG
    """
    y_true, y_pred = [], []

    if is_tumor:
        tumor_model.eval()
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs   = imgs.to(DEVICE)
                # Pass zeros for fMRI — not available at inference
                fmri_z = torch.zeros(imgs.size(0), 128, device=DEVICE)
                logits = tumor_model(imgs, fmri_z)
                preds  = logits.argmax(1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())
    else:
        face_model.eval()
        emotion_clf.eval()
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs   = imgs.to(DEVICE)
                feats  = face_model(imgs)
                logits = emotion_clf(feats)
                preds  = logits.argmax(1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc    = (y_true == y_pred).mean()

    print(f"\n  Overall accuracy: {acc*100:.2f}%\n")
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    ))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=class_names,
                yticklabels=class_names)
    label = "Tumor" if is_tumor else "Emotion"
    plt.title(f"{label} — Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")
    return acc


# ═════════════════════════════════════════════════════════════
#  EMOTION MODEL EVALUATION
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Emotion model evaluation")
print("=" * 60)

try:
    emotion_val    = ImageFolder(EMOTION_VAL_PATH, transform=emotion_tf)
    emotion_loader = DataLoader(emotion_val, batch_size=32,
                                shuffle=False, num_workers=0)

    print(f"\n  Path     : {EMOTION_VAL_PATH}")
    print(f"  Classes  : {emotion_val.classes}")
    print(f"  Images   : {len(emotion_val)}")

    if emotion_val.classes != EMOTION_CLASSES:
        print(f"\n  WARNING: class order mismatch!")
        print(f"  Dataset  : {emotion_val.classes}")
        print(f"  Expected : {EMOTION_CLASSES}")
        print(f"  Results may be incorrect.")
    else:
        print(f"  Class order matches EMOTION_CLASSES")

    print(f"\n  Per-class counts:")
    counts = Counter(emotion_val.targets)
    for idx, cls in enumerate(emotion_val.classes):
        bar = "█" * min(counts[idx] // 20, 40)
        print(f"    {cls:<12} {counts[idx]:>5}  {bar}")

    evaluate(
        dataloader  = emotion_loader,
        class_names = EMOTION_CLASSES,
        is_tumor    = False,
        save_path   = "confusion_matrix_emotion.png",
    )

except FileNotFoundError:
    print(f"\n  Emotion dataset not found at: {EMOTION_VAL_PATH}")
    print(f"  Make sure the path exists and contains class subfolders.")


# ═════════════════════════════════════════════════════════════
#  TUMOR MODEL EVALUATION
#  Uncomment once you have the tumor val dataset
# ═════════════════════════════════════════════════════════════

# print("\n" + "=" * 60)
# print("  Tumor model evaluation")
# print("=" * 60)
#
# try:
#     tumor_val    = ImageFolder(TUMOR_VAL_PATH, transform=tumor_tf)
#     tumor_loader = DataLoader(tumor_val, batch_size=32,
#                               shuffle=False, num_workers=0)
#
#     print(f"\n  Path     : {TUMOR_VAL_PATH}")
#     print(f"  Classes  : {tumor_val.classes}")
#     print(f"  Images   : {len(tumor_val)}")
#
#     if tumor_val.classes != TUMOR_CLASSES:
#         print(f"\n  WARNING: class order mismatch!")
#         print(f"  Dataset  : {tumor_val.classes}")
#         print(f"  Expected : {TUMOR_CLASSES}")
#     else:
#         print(f"  Class order matches TUMOR_CLASSES")
#
#     counts = Counter(tumor_val.targets)
#     print(f"\n  Per-class counts:")
#     for idx, cls in enumerate(tumor_val.classes):
#         bar = "█" * min(counts[idx] // 10, 40)
#         print(f"    {cls:<15} {counts[idx]:>5}  {bar}")
#
#     evaluate(
#         dataloader  = tumor_loader,
#         class_names = TUMOR_CLASSES,
#         is_tumor    = True,
#         save_path   = "confusion_matrix_tumor.png",
#     )
#
# except FileNotFoundError:
#     print(f"\n  Tumor dataset not found at: {TUMOR_VAL_PATH}")
#     print(f"  Download from Kaggle: masoudnickparvar/brain-tumor-mri-dataset")

print("\n  Done.\n")