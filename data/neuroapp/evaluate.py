"""
evaluate.py — Evaluate both Emotion and Tumor models.

Shows:
  - Overall accuracy
  - Per-class precision / recall / F1
  - Confusion matrix (saved as PNG)

Usage:
    cd C:\\Users\\HP\\ML_project\\data\\neuroapp
    python evaluate.py

Requirements:
    pip install scikit-learn seaborn matplotlib
"""

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from app import (
    emotion_model, emotion_clf, tumor_model,
    EMOTION_CLASSES, TUMOR_CLASSES,
    emotion_tf, tumor_tf,
    using_legacy_emotion, using_legacy_tumor,
    DEVICE, evaluate,
)

# ═════════════════════════════════════════════════════════════
#  PATHS — update tumor path once you download the dataset
# ═════════════════════════════════════════════════════════════

EMOTION_VAL_PATH = r"C:\Users\HP\ML_project\data\fer2013\test"

# Uncomment and set this once you download the tumor dataset from:
# https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
# TUMOR_VAL_PATH = r"C:\Users\HP\ML_project\data\tumor\Testing"

# ═════════════════════════════════════════════════════════════
#  EMOTION MODEL EVALUATION
# ═════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Loading emotion validation dataset...")
print("="*60)

try:
    emotion_val    = ImageFolder(EMOTION_VAL_PATH, transform=emotion_tf)
    emotion_loader = DataLoader(emotion_val, batch_size=32,
                                shuffle=False, num_workers=0)

    # Show dataset summary
    print(f"\n  Path     : {EMOTION_VAL_PATH}")
    print(f"  Classes  : {emotion_val.classes}")
    print(f"  Images   : {len(emotion_val)}")

    # Check class order matches EMOTION_CLASSES
    if emotion_val.classes != EMOTION_CLASSES:
        print(f"\n  ⚠️  WARNING: Class order mismatch!")
        print(f"  Dataset order  : {emotion_val.classes}")
        print(f"  Expected order : {EMOTION_CLASSES}")
        print(f"  Results may be incorrect — rename folders to match expected order.")
    else:
        print(f"  ✅ Class order matches EMOTION_CLASSES")

    # Per-class image counts
    print(f"\n  Per-class counts:")
    from collections import Counter
    counts = Counter(emotion_val.targets)
    for idx, cls in enumerate(emotion_val.classes):
        bar = "█" * min(counts[idx] // 20, 40)
        print(f"    {cls:<12} {counts[idx]:>5}  {bar}")

    # Run evaluation
    evaluate(
        model       = emotion_model,
        dataloader  = emotion_loader,
        class_names = EMOTION_CLASSES,
        device      = DEVICE,
        is_tumor    = False,
        save_path   = "confusion_matrix.png",
    )

except FileNotFoundError:
    print(f"\n  ❌ Emotion dataset not found at: {EMOTION_VAL_PATH}")
    print(f"  Make sure the path exists and has class subfolders.")

# ═════════════════════════════════════════════════════════════
#  TUMOR MODEL EVALUATION
#  Uncomment this block once you have the tumor dataset
# ═════════════════════════════════════════════════════════════

# print("\n" + "="*60)
# print("  Loading tumor validation dataset...")
# print("="*60)
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
#         print(f"\n  ⚠️  WARNING: Class order mismatch!")
#         print(f"  Dataset order  : {tumor_val.classes}")
#         print(f"  Expected order : {TUMOR_CLASSES}")
#     else:
#         print(f"  ✅ Class order matches TUMOR_CLASSES")
#
#     from collections import Counter
#     counts = Counter(tumor_val.targets)
#     print(f"\n  Per-class counts:")
#     for idx, cls in enumerate(tumor_val.classes):
#         bar = "█" * min(counts[idx] // 10, 40)
#         print(f"    {cls:<15} {counts[idx]:>5}  {bar}")
#
#     evaluate(
#         model       = tumor_model,
#         dataloader  = tumor_loader,
#         class_names = TUMOR_CLASSES,
#         device      = DEVICE,
#         is_tumor    = True,
#         save_path   = "confusion_matrix.png",
#     )
#
# except FileNotFoundError:
#     print(f"\n  ❌ Tumor dataset not found at: {TUMOR_VAL_PATH}")
#     print(f"  Download from: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")

print("\n  Done. Confusion matrix images saved in your neuroapp folder.\n")