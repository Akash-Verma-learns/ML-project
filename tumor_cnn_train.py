"""
tumor_cnn_train.py  —  CORRECTED for actual dataset structure
==============================================================

Figshare Brain Tumor dataset:
  figshare_brain/
    cvind.mat          <- official train/test split indices
    data/
      1.mat ... 3064.mat   <- cjdata.image, cjdata.label (1/2/3)

fMRI dataset (emotional faces task):
  fmri/
    onsetime/
      conmatrix_Run1.mat ... conmatrix_Run5.mat   <- connectivity matrices
      task-emotionalfaces_run-*.tsv               <- event timings
    Sub-01/ ... Sub-05/
      wrsub-0X_task-emotionalfaces_run-Y_bold.nii <- 4D BOLD volumes

Labels (Figshare):
  1 = meningioma
  2 = glioma
  3 = pituitary
  (no_tumor class does NOT exist in this dataset)

fMRI usage:
  Two feature types extracted per run:
    1. Mean BOLD activation  (4D nii -> temporal mean -> flatten -> subsample -> 64-d)
    2. Connectivity matrix   (conmatrix_RunX.mat -> upper triangle -> 64-d)
  Combined into 128-d vector per subject, z-scored across subjects.
  Since only 5 subjects exist, vectors are cycled across all 3064 MRI samples.

Run:
  pip install torch torchvision scipy nibabel scikit-learn matplotlib seaborn tqdm opencv-python
  python tumor_cnn_train.py
"""

import os, random, warnings
from pathlib import Path

import numpy as np
import scipy.io
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as tv_models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import cv2

warnings.filterwarnings("ignore")

# =====================================================
# GPU SETUP  -- enforced, with clear diagnostics
# =====================================================

if not torch.cuda.is_available():
    raise SystemError(
        "\n[ERROR] CUDA not available. Training requires a GPU.\n"
        "  - Make sure you installed the CUDA version of PyTorch:\n"
        "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
        "  - Check your driver: nvidia-smi\n"
        "  - Check PyTorch CUDA: python -c \"import torch; print(torch.version.cuda)\"\n"
    )

DEVICE = torch.device("cuda")

# RTX 3050 4GB optimizations
torch.backends.cudnn.benchmark    = True   # auto-tune convolution algorithms
torch.backends.cudnn.deterministic = False  # faster (allow non-deterministic ops)
torch.backends.cuda.matmul.allow_tf32 = True  # faster matmul on Ampere GPUs
torch.backends.cudnn.allow_tf32       = True  # faster convolutions on Ampere GPUs

# Print GPU info clearly
gpu_name   = torch.cuda.get_device_name(0)
vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"\n{'='*50}")
print(f"  GPU  : {gpu_name}")
print(f"  VRAM : {vram_total:.1f} GB")
print(f"  CUDA : {torch.version.cuda}")
print(f"{'='*50}\n")

# =====================================================
# PATHS  -- edit BASE_DIR to your project root
# =====================================================

BASE_DIR      = Path(r"C:\Users\HP\ML_project\data")
FIGSHARE_DIR  = BASE_DIR / "figshare_brain"
FIGSHARE_DATA = FIGSHARE_DIR / "data"
CVIND_PATH    = FIGSHARE_DIR / "cvind.mat"
FMRI_DIR      = BASE_DIR / "fmri"
ONSETIME_DIR  = FMRI_DIR / "onsetime"

CKPT_DIR  = Path(r"C:\Users\HP\ML_project")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL = CKPT_DIR / "tumor_cnn_best_new.pt"
LAST_MODEL = CKPT_DIR / "tumor_cnn_last_new.pt"

# =====================================================
# CONFIG  -- tuned for RTX 3050 4GB
# =====================================================

CFG = dict(
    img_size     = 224,
    batch_size   = 16,    # safe for 4GB VRAM (use 8 if OOM error)
    epochs       = 30,
    lr           = 1e-4,
    weight_decay = 1e-4,
    dropout      = 0.5,
    grad_clip    = 2.0,
    fmri_dim     = 128,   # 64 connectivity + 64 BOLD activation
    patience     = 8,
    seed         = 42,
)

# Figshare: exactly 3 classes, labels 1/2/3 -> 0/1/2
TUMOR_CLASSES = ["meningioma", "glioma", "pituitary"]
NUM_CLASSES   = 3

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# =====================================================
# fMRI FEATURE EXTRACTION
# Two sources: BOLD .nii volumes + connectivity .mat
# =====================================================

def load_connectivity_features(onsetime_dir: Path, target_dim: int = 64):
    """
    Load conmatrix_Run*.mat files from onsetime/.
    Each contains a square region-to-region connectivity matrix.
    Extract upper triangle, average across 5 runs, return target_dim-d vector.
    """
    vecs = []
    for run in range(1, 6):
        fpath = onsetime_dir / f"conmatrix_Run{run}.mat"
        if not fpath.exists():
            print(f"  [conn] {fpath.name} not found, skipping")
            continue
        try:
            mat  = scipy.io.loadmat(str(fpath))
            keys = [k for k in mat if not k.startswith("__")]
            if not keys:
                continue
            cm   = mat[keys[0]].astype(np.float32)
            if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
                print(f"  [conn] {fpath.name}: unexpected shape {cm.shape}, skipping")
                continue
            # Upper triangle (no diagonal)
            idx = np.triu_indices(cm.shape[0], k=1)
            vec = cm[idx]
            # Subsample to target_dim
            if len(vec) >= target_dim:
                step = len(vec) // target_dim
                vec  = vec[::step][:target_dim]
            else:
                vec = np.pad(vec, (0, target_dim - len(vec)))
            vecs.append(vec)
        except Exception as e:
            print(f"  [conn] Error loading {fpath.name}: {e}")

    if not vecs:
        print("  [conn] No connectivity matrices loaded — using zeros")
        return np.zeros(target_dim, dtype=np.float32)

    arr = np.stack(vecs)          # (n_runs, target_dim)
    print(f"  [conn] Loaded {len(vecs)} runs, averaged to {target_dim}-d vector")
    return arr.mean(axis=0).astype(np.float32)   # (target_dim,)


def load_bold_features(fmri_dir: Path, target_dim: int = 64):
    """
    Load 4D BOLD .nii volumes for each subject in Sub-XX folders.
    For each subject: temporal mean across all 5 runs -> spatial subsample.
    Returns: np.ndarray [n_subjects, target_dim]
    """
    subject_dirs = sorted([
        d for d in fmri_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith("sub")
    ])

    if not subject_dirs:
        print("  [BOLD] No Sub-XX folders found")
        return None

    all_vecs = []
    for sub_dir in subject_dirs:
        nii_files = sorted(list(sub_dir.glob("*.nii")) + list(sub_dir.glob("*.nii.gz")))
        if not nii_files:
            print(f"  [BOLD] {sub_dir.name}: no .nii files")
            continue

        run_vecs = []
        for nii_path in nii_files:
            try:
                img  = nib.load(str(nii_path))
                data = img.get_fdata(dtype=np.float32)
                # 4D (X,Y,Z,T) -> temporal mean -> 3D -> flatten
                if data.ndim == 4:
                    data = data.mean(axis=-1)
                vec = data.flatten()
                # Uniform subsample to target_dim
                if len(vec) >= target_dim:
                    idx = np.linspace(0, len(vec) - 1, target_dim, dtype=int)
                    vec = vec[idx]
                else:
                    vec = np.pad(vec, (0, target_dim - len(vec)))
                run_vecs.append(vec.astype(np.float32))
            except Exception as e:
                print(f"  [BOLD] Error {nii_path.name}: {e}")

        if run_vecs:
            # Average across runs for this subject
            sub_vec = np.stack(run_vecs).mean(axis=0)
            all_vecs.append(sub_vec)
            print(f"  [BOLD] {sub_dir.name}: {len(run_vecs)} runs -> {target_dim}-d")

    if not all_vecs:
        print("  [BOLD] No valid volumes found")
        return None

    return np.stack(all_vecs, axis=0).astype(np.float32)  # (n_subjects, target_dim)


def extract_fmri_features(fmri_dir: Path, fmri_dim: int = 128):
    """
    Combine connectivity (64-d) + BOLD activation (64-d) -> fmri_dim-d per subject.
    Z-score across subjects.
    Returns: np.ndarray [n_subjects, fmri_dim] or None
    """
    half = fmri_dim // 2

    print("\n[fMRI] Extracting connectivity features...")
    conn_vec = load_connectivity_features(ONSETIME_DIR, target_dim=half)

    print("[fMRI] Extracting BOLD activation features...")
    bold_mat = load_bold_features(fmri_dir, target_dim=half)

    if bold_mat is None:
        print("[fMRI] No BOLD data - fMRI branch will be disabled")
        return None

    n_sub = bold_mat.shape[0]

    # Broadcast single connectivity vec to all subjects, concat with BOLD
    conn_bc  = np.tile(conn_vec, (n_sub, 1))              # (n_sub, half)
    combined = np.concatenate([conn_bc, bold_mat], axis=1) # (n_sub, fmri_dim)

    # Z-score per feature across subjects
    scaler   = StandardScaler()
    combined = scaler.fit_transform(combined).astype(np.float32)

    print(f"[fMRI] Final: {combined.shape} ({n_sub} subjects x {fmri_dim}-d)")
    return combined


# Run fMRI extraction
fmri_features = extract_fmri_features(FMRI_DIR, fmri_dim=CFG["fmri_dim"])
USE_FMRI      = fmri_features is not None
N_SUBJECTS    = fmri_features.shape[0] if USE_FMRI else 0

if USE_FMRI:
    print(f"[fMRI] Will cycle {N_SUBJECTS} subject vectors across 3064 MRI samples")

# =====================================================
# FIGSHARE .MAT LOADER
# =====================================================

def load_mat_sample(mat_path: Path):
    """
    Load one Figshare .mat file.
    Supports both scipy (MATLAB < v7.3) and h5py (MATLAB v7.3 HDF5).
    Returns: (image_uint8 ndarray HxW, label_0based int)
    cjdata.label: 1=meningioma, 2=glioma, 3=pituitary -> 0,1,2
    """
    import h5py

    # Try scipy first (old format)
    try:
        mat   = scipy.io.loadmat(str(mat_path))
        data  = mat["cjdata"][0, 0]
        image = data["image"].astype(np.float32)
        label = int(data["label"].flat[0])

    except NotImplementedError:
        # MATLAB v7.3 HDF5 format
        with h5py.File(str(mat_path), "r") as f:
            cjdata = f["cjdata"]
            # image is stored transposed in HDF5 matlab files
            image  = np.array(cjdata["image"]).T.astype(np.float32)
            label  = int(np.array(cjdata["label"]).flat[0])

    # Normalize to 0-255 uint8
    lo, hi = image.min(), image.max()
    if hi > lo:
        image = (image - lo) / (hi - lo) * 255.0
    image = image.astype(np.uint8)

    return image, label - 1   # 0-based label

# =====================================================
# DATASET
# =====================================================

class FigshareDataset(Dataset):
    def __init__(self, file_indices, data_dir, transform=None, fmri_feats=None):
        self.data_dir   = data_dir
        self.transform  = transform
        self.fmri_feats = fmri_feats
        self.n_sub      = fmri_feats.shape[0] if fmri_feats is not None else 0

        self.samples = []
        for idx in file_indices:
            p = data_dir / f"{idx}.mat"
            if p.exists():
                self.samples.append(idx)

        print(f"  Dataset ready: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_np, lbl = load_mat_sample(self.data_dir / f"{self.samples[i]}.mat")

        pil = Image.fromarray(img_np).convert("RGB")
        img_tensor = self.transform(pil) if self.transform else T.ToTensor()(pil)

        # Cycle fMRI features over subjects
        if self.fmri_feats is not None:
            fmri_vec = torch.tensor(self.fmri_feats[i % self.n_sub], dtype=torch.float32)
        else:
            fmri_vec = torch.zeros(CFG["fmri_dim"], dtype=torch.float32)

        return img_tensor, lbl, fmri_vec

# =====================================================
# LOAD cvind.mat  -- official train/test split
# =====================================================

def load_split_indices(cvind_path: Path, n_total: int = 3064):
    """
    cvind.mat contains cross-validation fold assignments.
    Supports both old scipy format AND new MATLAB v7.3 HDF5 format.
    Standard Figshare practice: use fold 1 as test, rest as train.
    Falls back to random 85/15 if file missing.
    Returns: (train_indices, test_indices) -- 1-based file numbers
    """
    import h5py

    all_idx = list(range(1, n_total + 1))

    if not cvind_path.exists():
        print("[Split] cvind.mat not found -- random 85/15 split")
        random.shuffle(all_idx)
        cut = int(len(all_idx) * 0.85)
        return all_idx[:cut], all_idx[cut:]

    # Try scipy first (MATLAB < v7.3)
    try:
        mat   = scipy.io.loadmat(str(cvind_path))
        keys  = [k for k in mat if not k.startswith("__")]
        print(f"[Split] scipy format | keys: {keys}")
        cvind = mat[keys[0]]

        if cvind.dtype == object:
            test_idx  = cvind[0, 0].flatten().astype(int).tolist()
            train_idx = [i for i in all_idx if i not in set(test_idx)]
        else:
            cvind     = cvind.flatten().astype(int)
            test_idx  = [i + 1 for i, f in enumerate(cvind) if f == 1]
            train_idx = [i + 1 for i, f in enumerate(cvind) if f != 1]

        print(f"[Split] train={len(train_idx)}  test={len(test_idx)}")
        return train_idx, test_idx

    except NotImplementedError:
        pass  # v7.3 HDF5 file -- fall through to h5py

    # h5py reader for MATLAB v7.3 HDF5 format
    print("[Split] MATLAB v7.3 detected -- using h5py")
    with h5py.File(str(cvind_path), "r") as f:
        keys = list(f.keys())
        print(f"[Split] h5py keys: {keys}")

        # cvind is typically stored as a 2D array or cell references
        raw = f[keys[0]]

        # Case 1: direct numeric dataset (n_samples,) or (1, n_samples)
        if isinstance(raw, h5py.Dataset):
            cvind = np.array(raw).flatten().astype(int)
            print(f"[Split] Dataset shape: {cvind.shape}  unique values: {np.unique(cvind)}")

            # If values are fold numbers (e.g. 1-5): fold 1 = test
            if cvind.max() <= 20:
                test_idx  = [i + 1 for i, f in enumerate(cvind) if f == 1]
                train_idx = [i + 1 for i, f in enumerate(cvind) if f != 1]
            else:
                # Values are actual 1-based file indices for one fold
                test_idx  = cvind.tolist()
                train_idx = [i for i in all_idx if i not in set(test_idx)]

        # Case 2: cell array of HDF5 references
        elif isinstance(raw, h5py.Group):
            # Each item in the group is a fold's index array
            fold_arrays = []
            for k in raw.keys():
                fold_arrays.append(np.array(raw[k]).flatten().astype(int))
            # Use fold 0 as test set
            test_idx  = fold_arrays[0].tolist()
            train_idx = [i for i in all_idx if i not in set(test_idx)]
            print(f"[Split] Cell group with {len(fold_arrays)} folds")

        else:
            print("[Split] Unknown cvind format -- falling back to random split")
            random.shuffle(all_idx)
            cut = int(len(all_idx) * 0.85)
            return all_idx[:cut], all_idx[cut:]

    print(f"[Split] train={len(train_idx)}  test={len(test_idx)}")
    return train_idx, test_idx


print("\n[Data] Loading split indices...")
train_idx, test_idx = load_split_indices(CVIND_PATH)

# =====================================================
# TRANSFORMS
# =====================================================

train_tf = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(p=0.2),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_tf = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# =====================================================
# BUILD DATASETS
# =====================================================

print("\n[Data] Building datasets...")
train_set = FigshareDataset(train_idx, FIGSHARE_DATA, train_tf, fmri_features)
test_set  = FigshareDataset(test_idx,  FIGSHARE_DATA, test_tf,  fmri_features)

# Weighted sampler for class balance
print("[Data] Computing class distribution...")
all_labels = []
for idx in train_set.samples:
    _, lbl = load_mat_sample(FIGSHARE_DATA / f"{idx}.mat")
    all_labels.append(lbl)

class_counts  = np.bincount(all_labels, minlength=NUM_CLASSES)
class_weights = 1.0 / (class_counts + 1e-8)
sample_wts    = torch.tensor([class_weights[l] for l in all_labels], dtype=torch.float32)
sampler       = WeightedRandomSampler(sample_wts, len(sample_wts), replacement=True)

print(f"[Data] Counts: { {TUMOR_CLASSES[i]: int(class_counts[i]) for i in range(NUM_CLASSES)} }")

train_loader = DataLoader(train_set, batch_size=CFG["batch_size"],
                          sampler=sampler, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set, batch_size=CFG["batch_size"],
                          shuffle=False, num_workers=0, pin_memory=True)

# =====================================================
# MODEL
# =====================================================

class MRIEncoder(nn.Module):
    """EfficientNet-B0 -> [B, 1280]"""
    def __init__(self, dropout=0.5):
        super().__init__()
        base          = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.drop     = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.drop(x.view(x.size(0), -1))


class FMRIEncoder(nn.Module):
    """
    Encodes 128-d fMRI vector (64 connectivity + 64 BOLD) -> 64-d
    """
    def __init__(self, in_dim: int, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BrainTumorCNN(nn.Module):
    """
    Late-fusion:
      MRI  -> EfficientNet-B0 -> 1280-d
      fMRI -> MLP              ->   64-d  (optional)
      Concat -> BN -> FC(512) -> FC(256) -> FC(3)
    """
    def __init__(self, use_fmri=False, fmri_in_dim=128,
                 num_classes=3, dropout=0.5):
        super().__init__()
        self.use_fmri = use_fmri
        self.mri_enc  = MRIEncoder(dropout=dropout)

        fused = 1280
        if use_fmri:
            self.fmri_enc = FMRIEncoder(fmri_in_dim, dropout=dropout)
            fused += 64

        self.classifier = nn.Sequential(
            nn.Linear(fused, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, fmri=None):
        feat = self.mri_enc(img)
        if self.use_fmri and fmri is not None:
            feat = torch.cat([feat, self.fmri_enc(fmri)], dim=1)
        return self.classifier(feat)


model = BrainTumorCNN(
    use_fmri    = USE_FMRI,
    fmri_in_dim = CFG["fmri_dim"],
    num_classes = NUM_CLASSES,
    dropout     = CFG["dropout"],
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[Model] BrainTumorCNN | use_fmri={USE_FMRI} | params={n_params:,}")

# =====================================================
# LOSS / OPTIMIZER / SCHEDULER
# =====================================================

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# Meningioma hardest -> slightly higher weight
class_wt  = torch.tensor([1.4, 1.2, 1.0], device=DEVICE)
criterion = FocalLoss(weight=class_wt, gamma=1.5)

optimizer = torch.optim.AdamW(model.parameters(),
                               lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

# =====================================================
# TRAINING
# =====================================================

best_acc, no_improve = 0.0, 0
train_losses, val_accs = [], []

print("\n" + "=" * 60)
print(f"  Training ({CFG['epochs']} epochs, patience={CFG['patience']})")
print("=" * 60)

for epoch in range(1, CFG["epochs"] + 1):
    model.train()
    run_loss, correct, total = 0.0, 0, 0

    bar = tqdm(train_loader, desc=f"Ep {epoch:02d}/{CFG['epochs']}", leave=False, ncols=80)
    for imgs, labels, fmri_vecs in bar:
        imgs      = imgs.to(DEVICE)
        labels    = labels.to(DEVICE)
        fmri_vecs = fmri_vecs.to(DEVICE) if USE_FMRI else None

        optimizer.zero_grad()
        try:
            logits = model(imgs, fmri_vecs)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\n[OOM] Out of VRAM! Reduce batch_size in CFG (currently {CFG['batch_size']}).")
            print(f"      Try batch_size=8 and restart.\n")
            raise

        run_loss += loss.item()
        preds     = logits.argmax(1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)
        bar.set_postfix(loss=f"{loss.item():.3f}")

    scheduler.step()
    train_loss = run_loss / len(train_loader)
    train_acc  = 100 * correct / total
    train_losses.append(train_loss)

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels, fmri_vecs in test_loader:
            imgs      = imgs.to(DEVICE)
            labels    = labels.to(DEVICE)
            fmri_vecs = fmri_vecs.to(DEVICE) if USE_FMRI else None
            logits    = model(imgs, fmri_vecs)
            preds     = logits.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = 100 * val_correct / val_total
    val_accs.append(val_acc)

    if val_acc > best_acc:
        best_acc, no_improve = val_acc, 0
        torch.save(model.state_dict(), str(BEST_MODEL))
        flag = " <- best"
    else:
        no_improve += 1
        flag = ""

    vram_used = torch.cuda.memory_reserved(0) / 1024**3
    print(f"Ep {epoch:02d}/{CFG['epochs']}  "
          f"Loss: {train_loss:.4f}  "
          f"Train: {train_acc:.1f}%  "
          f"Val: {val_acc:.1f}%  "
          f"LR: {scheduler.get_last_lr()[0]:.2e}  "
          f"VRAM: {vram_used:.1f}GB{flag}")

    if no_improve >= CFG["patience"]:
        print(f"\n[Early stop] {CFG['patience']} epochs without improvement.")
        break

torch.save(model.state_dict(), str(LAST_MODEL))
print(f"\nBest val accuracy: {best_acc:.2f}%")

# =====================================================
# EVALUATION
# =====================================================

model.load_state_dict(torch.load(str(BEST_MODEL), map_location=DEVICE))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels, fmri_vecs in test_loader:
        imgs      = imgs.to(DEVICE)
        fmri_vecs = fmri_vecs.to(DEVICE) if USE_FMRI else None
        preds     = model(imgs, fmri_vecs).argmax(1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=TUMOR_CLASSES, zero_division=0))

cm  = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=TUMOR_CLASSES, yticklabels=TUMOR_CLASSES, ax=axes[0])
axes[0].set_title("Confusion Matrix"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
axes[0].tick_params(axis="x", rotation=30)

ep = range(1, len(train_losses) + 1)
axes[1].plot(ep, train_losses, "#e74c3c", linewidth=2, label="Train Loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss", color="#e74c3c")
ax2 = axes[1].twinx()
ax2.plot(ep, val_accs, "#3498db", linewidth=2, label="Val Acc %")
ax2.set_ylabel("Accuracy %", color="#3498db")
axes[1].set_title("Learning Curves"); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("tumor_training_results.png", dpi=150)
plt.show()

# =====================================================
# GRAD-CAM  -- 3-panel matching app.py exactly
# INFERNO colormap + percentile threshold + contours
# =====================================================

class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = []
        self.activations = []
        last_conv = model.mri_enc.features[-1]
        last_conv.register_forward_hook(
            lambda m, i, o: self.activations.__setitem__(0, o.detach()) or None)
        last_conv.register_full_backward_hook(
            lambda m, gi, go: self.gradients.__setitem__(0, go[0].detach()) or None)
        self.activations = [None]
        self.gradients   = [None]

    def generate(self, img_tensor, class_idx=None):
        self.model.zero_grad()
        logits = self.model(img_tensor, fmri=None)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        logits[0, class_idx].backward()

        grad = self.gradients[0].squeeze(0)
        act  = self.activations[0].squeeze(0)
        wts  = F.relu(grad).mean(dim=(1, 2))
        cam  = (wts[:, None, None] * act).sum(0)
        cam  = F.relu(cam).cpu().numpy()
        cam  = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam, class_idx


def gradcam_3panel(orig_rgb, cam_np):
    """Build overlay / pure heatmap / contour -- matches app.py"""
    h, w   = orig_rgb.shape[:2]
    cam_r  = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_CUBIC)

    # Percentile threshold (suppress bottom 30%)
    thresh    = np.percentile(cam_r, 70)
    cam_t     = np.where(cam_r >= thresh, cam_r, cam_r * 0.2)
    cam_t     = (cam_t - cam_t.min()) / (cam_t.max() + 1e-8)

    cam_u8    = (cam_t * 255).astype(np.uint8)
    heat      = cv2.applyColorMap(cam_u8, cv2.COLORMAP_INFERNO)
    heat_rgb  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    alpha_map = (0.35 + 0.35 * cam_t)[:, :, np.newaxis]

    overlay   = (orig_rgb * (1 - alpha_map) + heat_rgb * alpha_map).astype(np.uint8)
    heatmap_v = (np.zeros_like(orig_rgb) * (1 - alpha_map) + heat_rgb * alpha_map).astype(np.uint8)

    contour_img     = overlay.copy()
    mask            = (cam_t > 0.5).astype(np.uint8) * 255
    contours, _     = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 180), 2)

    return overlay, heatmap_v, contour_img


print("\n[Grad-CAM] Generating 3-panel saliency maps...")
cam_engine = GradCAM(model)
model.eval()

# One sample per class from test set
class_samples = {}
for file_idx in test_set.samples:
    _, lbl = load_mat_sample(FIGSHARE_DATA / f"{file_idx}.mat")
    if lbl not in class_samples:
        class_samples[lbl] = file_idx
    if len(class_samples) == NUM_CLASSES:
        break

fig, axes = plt.subplots(3, NUM_CLASSES, figsize=(5 * NUM_CLASSES, 15))
row_titles = ["Original", "Overlay (INFERNO)", "Contour"]

for cls_idx in range(NUM_CLASSES):
    file_idx = class_samples.get(cls_idx)
    if file_idx is None:
        for r in range(3):
            axes[r, cls_idx].axis("off")
        continue

    img_np, _ = load_mat_sample(FIGSHARE_DATA / f"{file_idx}.mat")
    orig_rgb  = np.stack([img_np] * 3, axis=-1)

    pil = Image.fromarray(img_np).convert("RGB")
    inp = test_tf(pil).unsqueeze(0).to(DEVICE).requires_grad_(True)

    cam, _ = cam_engine.generate(inp, class_idx=cls_idx)
    overlay, heatmap_v, contour_img = gradcam_3panel(orig_rgb, cam)

    for r, (panel, title) in enumerate(zip(
            [img_np, overlay, contour_img], row_titles)):
        axes[r, cls_idx].imshow(panel, cmap="gray" if r == 0 else None)
        axes[r, cls_idx].set_title(f"{title}\n{TUMOR_CLASSES[cls_idx]}", fontsize=10)
        axes[r, cls_idx].axis("off")

plt.suptitle("Grad-CAM 3-Panel — Brain Tumor CNN", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("tumor_gradcam_3panel.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n[Done]")
print(f"  Best checkpoint      : {BEST_MODEL}")
print(f"  Training plots       : tumor_training_results.png")
print(f"  Grad-CAM 3-panel     : tumor_gradcam_3panel.png")
print(f"\n  Copy to NeuroApp:")
print(f"  copy {BEST_MODEL} neuroapp\\checkpoints\\tumor_cnn_best.pt")