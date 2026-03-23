"""
tumor_cnn_train.py  —  TRANSFORMER FUSION VERSION
==============================================================
CHANGE FROM ORIGINAL:
  OLD: MRI features + fMRI features → torch.cat → Linear layers
  NEW: MRI features + fMRI features → Multi-Head Cross-Attention
                                       (N stacked layers) → classifier

Everything else is identical:
  - EfficientNet-B0 CNN backbone (unchanged)
  - FMRIEncoder MLP (unchanged)
  - Dataset loading, transforms, training loop (unchanged)

Architecture of new fusion block:
  MRI  (1280-d) → project → (256-d) ┐
                                      Multi-Head Cross-Attn  ← layer 1
  fMRI  (64-d)  → project → (256-d) ┘      ↓
                                      Multi-Head Cross-Attn  ← layer 2
                                            ↓
                                      Multi-Head Cross-Attn  ← layer 3
                                            ↓
                                      LayerNorm → FC(128) → FC(3)
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
# GPU SETUP
# =====================================================

if not torch.cuda.is_available():
    raise SystemError(
        "\n[ERROR] CUDA not available. Training requires a GPU.\n"
        "  pip install torch torchvision --index-url "
        "https://download.pytorch.org/whl/cu121\n"
    )

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.deterministic    = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

gpu_name   = torch.cuda.get_device_name(0)
vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"\n{'='*50}")
print(f"  GPU  : {gpu_name}")
print(f"  VRAM : {vram_total:.1f} GB")
print(f"  CUDA : {torch.version.cuda}")
print(f"{'='*50}\n")

# =====================================================
# PATHS
# =====================================================

BASE_DIR      = Path(r"C:\Users\HP\ML_project\data")
FIGSHARE_DIR  = BASE_DIR / "figshare_brain"
FIGSHARE_DATA = FIGSHARE_DIR / "data"
CVIND_PATH    = FIGSHARE_DIR / "cvind.mat"
FMRI_DIR      = BASE_DIR / "fmri"
ONSETIME_DIR  = FMRI_DIR / "onsetime"

CKPT_DIR   = Path(r"C:\Users\HP\ML_project")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL = CKPT_DIR / "tumor_transformer_best.pt"   # new name — different arch
LAST_MODEL = CKPT_DIR / "tumor_transformer_last.pt"

# =====================================================
# CONFIG
# =====================================================

CFG = dict(
    img_size      = 224,
    batch_size    = 16,
    epochs        = 30,
    lr            = 1e-4,
    weight_decay  = 1e-4,
    dropout       = 0.5,
    grad_clip     = 2.0,
    fmri_dim      = 128,
    patience      = 8,
    seed          = 42,

    # ── Transformer fusion hyperparams ──────────────
    fusion_dim    = 256,   # project MRI+fMRI to this dim before attention
    num_heads     = 8,     # attention heads (fusion_dim must be divisible)
    num_layers    = 3,     # how many stacked cross-attention layers
    ffn_dim       = 512,   # feed-forward dim inside each transformer layer
    attn_dropout  = 0.1,   # dropout inside attention
)

TUMOR_CLASSES = ["meningioma", "glioma", "pituitary"]
NUM_CLASSES   = 3

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])

print(f"Device: {DEVICE}")


# =====================================================
# fMRI FEATURE EXTRACTION  (unchanged)
# =====================================================

def load_connectivity_features(onsetime_dir: Path, target_dim: int = 64):
    vecs = []
    for run in range(1, 6):
        fpath = onsetime_dir / f"conmatrix_Run{run}.mat"
        if not fpath.exists():
            continue
        try:
            mat  = scipy.io.loadmat(str(fpath))
            keys = [k for k in mat if not k.startswith("__")]
            if not keys:
                continue
            cm  = mat[keys[0]].astype(np.float32)
            if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
                continue
            idx = np.triu_indices(cm.shape[0], k=1)
            vec = cm[idx]
            if len(vec) >= target_dim:
                step = len(vec) // target_dim
                vec  = vec[::step][:target_dim]
            else:
                vec = np.pad(vec, (0, target_dim - len(vec)))
            vecs.append(vec)
        except Exception as e:
            print(f"  [conn] Error {fpath.name}: {e}")

    if not vecs:
        return np.zeros(target_dim, dtype=np.float32)
    return np.stack(vecs).mean(axis=0).astype(np.float32)


def load_bold_features(fmri_dir: Path, target_dim: int = 64):
    subject_dirs = sorted([
        d for d in fmri_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith("sub")
    ])
    if not subject_dirs:
        return None

    all_vecs = []
    for sub_dir in subject_dirs:
        nii_files = sorted(
            list(sub_dir.glob("*.nii")) + list(sub_dir.glob("*.nii.gz"))
        )
        if not nii_files:
            continue
        run_vecs = []
        for nii_path in nii_files:
            try:
                img  = nib.load(str(nii_path))
                data = img.get_fdata(dtype=np.float32)
                if data.ndim == 4:
                    data = data.mean(axis=-1)
                vec = data.flatten()
                if len(vec) >= target_dim:
                    idx = np.linspace(0, len(vec) - 1, target_dim, dtype=int)
                    vec = vec[idx]
                else:
                    vec = np.pad(vec, (0, target_dim - len(vec)))
                run_vecs.append(vec.astype(np.float32))
            except Exception as e:
                print(f"  [BOLD] Error {nii_path.name}: {e}")
        if run_vecs:
            all_vecs.append(np.stack(run_vecs).mean(axis=0))

    if not all_vecs:
        return None
    return np.stack(all_vecs, axis=0).astype(np.float32)


def extract_fmri_features(fmri_dir: Path, fmri_dim: int = 128):
    half = fmri_dim // 2
    print("\n[fMRI] Extracting connectivity features...")
    conn_vec = load_connectivity_features(ONSETIME_DIR, target_dim=half)
    print("[fMRI] Extracting BOLD activation features...")
    bold_mat = load_bold_features(fmri_dir, target_dim=half)

    if bold_mat is None:
        print("[fMRI] No BOLD data — fMRI branch disabled")
        return None

    n_sub    = bold_mat.shape[0]
    conn_bc  = np.tile(conn_vec, (n_sub, 1))
    combined = np.concatenate([conn_bc, bold_mat], axis=1)
    scaler   = StandardScaler()
    combined = scaler.fit_transform(combined).astype(np.float32)
    print(f"[fMRI] Final: {combined.shape}")
    return combined


fmri_features = extract_fmri_features(FMRI_DIR, fmri_dim=CFG["fmri_dim"])
USE_FMRI      = fmri_features is not None
N_SUBJECTS    = fmri_features.shape[0] if USE_FMRI else 0


# =====================================================
# DATASET  (unchanged)
# =====================================================

def load_mat_sample(mat_path: Path):
    import h5py
    try:
        mat   = scipy.io.loadmat(str(mat_path))
        data  = mat["cjdata"][0, 0]
        image = data["image"].astype(np.float32)
        label = int(data["label"].flat[0])
    except NotImplementedError:
        with h5py.File(str(mat_path), "r") as f:
            cjdata = f["cjdata"]
            image  = np.array(cjdata["image"]).T.astype(np.float32)
            label  = int(np.array(cjdata["label"]).flat[0])
    lo, hi = image.min(), image.max()
    if hi > lo:
        image = (image - lo) / (hi - lo) * 255.0
    return image.astype(np.uint8), label - 1


class FigshareDataset(Dataset):
    def __init__(self, file_indices, data_dir, transform=None, fmri_feats=None):
        self.data_dir   = data_dir
        self.transform  = transform
        self.fmri_feats = fmri_feats
        self.n_sub      = fmri_feats.shape[0] if fmri_feats is not None else 0
        self.samples    = [i for i in file_indices if (data_dir / f"{i}.mat").exists()]
        print(f"  Dataset ready: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_np, lbl = load_mat_sample(self.data_dir / f"{self.samples[i]}.mat")
        pil         = Image.fromarray(img_np).convert("RGB")
        img_tensor  = self.transform(pil) if self.transform else T.ToTensor()(pil)
        if self.fmri_feats is not None:
            fmri_vec = torch.tensor(
                self.fmri_feats[i % self.n_sub], dtype=torch.float32)
        else:
            fmri_vec = torch.zeros(CFG["fmri_dim"], dtype=torch.float32)
        return img_tensor, lbl, fmri_vec


def load_split_indices(cvind_path: Path, n_total: int = 3064):
    import h5py
    all_idx = list(range(1, n_total + 1))
    if not cvind_path.exists():
        random.shuffle(all_idx)
        cut = int(len(all_idx) * 0.85)
        return all_idx[:cut], all_idx[cut:]
    try:
        mat   = scipy.io.loadmat(str(cvind_path))
        keys  = [k for k in mat if not k.startswith("__")]
        cvind = mat[keys[0]]
        if cvind.dtype == object:
            test_idx  = cvind[0, 0].flatten().astype(int).tolist()
            train_idx = [i for i in all_idx if i not in set(test_idx)]
        else:
            cvind     = cvind.flatten().astype(int)
            test_idx  = [i + 1 for i, f in enumerate(cvind) if f == 1]
            train_idx = [i + 1 for i, f in enumerate(cvind) if f != 1]
        return train_idx, test_idx
    except NotImplementedError:
        pass
    with h5py.File(str(cvind_path), "r") as f:
        keys = list(f.keys())
        raw  = f[keys[0]]
        if isinstance(raw, h5py.Dataset):
            cvind = np.array(raw).flatten().astype(int)
            if cvind.max() <= 20:
                test_idx  = [i + 1 for i, f in enumerate(cvind) if f == 1]
                train_idx = [i + 1 for i, f in enumerate(cvind) if f != 1]
            else:
                test_idx  = cvind.tolist()
                train_idx = [i for i in all_idx if i not in set(test_idx)]
        else:
            fold_arrays = [np.array(raw[k]).flatten().astype(int) for k in raw.keys()]
            test_idx    = fold_arrays[0].tolist()
            train_idx   = [i for i in all_idx if i not in set(test_idx)]
    return train_idx, test_idx


# =====================================================
# TRANSFORMS  (unchanged)
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

# Build datasets
print("\n[Data] Loading split indices...")
train_idx, test_idx = load_split_indices(CVIND_PATH)

print("\n[Data] Building datasets...")
train_set = FigshareDataset(train_idx, FIGSHARE_DATA, train_tf, fmri_features)
test_set  = FigshareDataset(test_idx,  FIGSHARE_DATA, test_tf,  fmri_features)

print("[Data] Computing class distribution...")
all_labels    = [load_mat_sample(FIGSHARE_DATA / f"{i}.mat")[1] for i in train_set.samples]
class_counts  = np.bincount(all_labels, minlength=NUM_CLASSES)
class_weights = 1.0 / (class_counts + 1e-8)
sample_wts    = torch.tensor([class_weights[l] for l in all_labels], dtype=torch.float32)
sampler       = WeightedRandomSampler(sample_wts, len(sample_wts), replacement=True)

print(f"[Data] Counts: { {TUMOR_CLASSES[i]: int(class_counts[i]) for i in range(NUM_CLASSES)} }")

train_loader = DataLoader(train_set, batch_size=CFG["batch_size"],
                          sampler=sampler, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=CFG["batch_size"],
                          shuffle=False, num_workers=0, pin_memory=True)


# =====================================================
# ENCODERS  (CNN — unchanged from original)
# =====================================================

class MRIEncoder(nn.Module):
    """EfficientNet-B0 CNN backbone → [B, 1280]  (unchanged)"""
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
        return self.drop(x.view(x.size(0), -1))   # (B, 1280)


class FMRIEncoder(nn.Module):
    """128-d fMRI vector → 64-d  (unchanged)"""
    def __init__(self, in_dim: int = 128, dropout: float = 0.4):
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
        return self.net(x)   # (B, 64)


# =====================================================
# ★ NEW: MULTI-HEAD CROSS-ATTENTION FUSION TRANSFORMER
# =====================================================

class CrossAttentionLayer(nn.Module):
    """
    One layer of cross-attention + self-attention + feed-forward.

    Step 1 — Cross-Attention:
        MRI token QUERIES the fMRI token.
        "Given what the CNN sees in the MRI, what fMRI info matters?"
        The answer changes per-sample — unlike the old scalar weights.

    Step 2 — Self-Attention:
        The updated MRI token attends to itself (refines its own representation).

    Step 3 — Feed-Forward Network:
        Standard transformer FFN with GELU activation.

    Each step has a residual connection + LayerNorm (Pre-LN style —
    more stable training than Post-LN).
    """
    def __init__(self, dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()

        # ── Cross-attention: MRI (query) ← fMRI (key, value) ─────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,   # (B, seq, dim) instead of (seq, B, dim)
        )
        self.norm_cross = nn.LayerNorm(dim)

        # ── Self-attention: MRI token refines itself ───────────────────────
        self.self_attn  = nn.MultiheadAttention(
            embed_dim   = dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm_self  = nn.LayerNorm(dim)

        # ── Feed-forward network ───────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),               # smoother than ReLU, standard in transformers
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, mri_tok, fmri_tok):
        """
        Args:
            mri_tok  : (B, 1, dim)  — MRI feature token
            fmri_tok : (B, 1, dim)  — fMRI feature token
        Returns:
            mri_tok  : (B, 1, dim)  — updated MRI token (fMRI info absorbed)
        """
        # Step 1: Cross-attention (Pre-LN: normalize before attention)
        mri_norm = self.norm_cross(mri_tok)
        attended, attn_weights = self.cross_attn(
            query = mri_norm,    # MRI asks the question
            key   = fmri_tok,    # fMRI provides context
            value = fmri_tok,
        )
        mri_tok = mri_tok + attended    # residual connection

        # Step 2: Self-attention
        mri_norm = self.norm_self(mri_tok)
        refined, _ = self.self_attn(mri_norm, mri_norm, mri_norm)
        mri_tok  = mri_tok + refined    # residual connection

        # Step 3: Feed-forward
        mri_norm = self.norm_ffn(mri_tok)
        mri_tok  = mri_tok + self.ffn(mri_norm)   # residual connection

        return mri_tok, attn_weights


class TransformerFusion(nn.Module):
    """
    Multi-layer cross-attention fusion block.

    Replaces the old:
        torch.cat([mri_1280, fmri_64]) → Linear(1344→512) → Linear(512→256) → Linear(256→3)

    With:
        project(mri_1280  → 256)  ┐
                                   CrossAttentionLayer × N  → LayerNorm → FC(128) → FC(3)
        project(fmri_64   → 256)  ┘

    Why stacking layers helps:
        Layer 1: MRI absorbs coarse fMRI structure info
        Layer 2: Refines — which fMRI regions relate to the tumor location?
        Layer 3: Fine-grained — subtle correlations between modalities
    """
    def __init__(self, mri_dim: int = 1280, fmri_dim: int = 64,
                 fusion_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 3, ffn_dim: int = 512,
                 num_classes: int = 3, dropout: float = 0.1):
        super().__init__()

        # Project both modalities to the same fusion_dim
        # (required so they can interact in the same attention space)
        self.mri_proj  = nn.Sequential(
            nn.Linear(mri_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.fmri_proj = nn.Sequential(
            nn.Linear(fmri_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Stack N cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                dim      = fusion_dim,
                num_heads= num_heads,
                ffn_dim  = ffn_dim,
                dropout  = dropout,
            )
            for _ in range(num_layers)
        ])

        # Final classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Store attention weights for visualization / interpretability
        self.last_attn_weights = None

    def forward(self, mri_feat, fmri_feat):
        """
        Args:
            mri_feat  : (B, 1280)  from EfficientNet-B0
            fmri_feat : (B, 64)    from FMRIEncoder
        Returns:
            logits    : (B, num_classes)
        """
        # Project to fusion space and add sequence dimension
        mri_tok  = self.mri_proj(mri_feat).unsqueeze(1)    # (B, 1, fusion_dim)
        fmri_tok = self.fmri_proj(fmri_feat).unsqueeze(1)  # (B, 1, fusion_dim)

        # Pass through N stacked cross-attention layers
        attn_weights = None
        for layer in self.layers:
            mri_tok, attn_weights = layer(mri_tok, fmri_tok)

        # Save for optional visualization
        self.last_attn_weights = attn_weights

        # Remove sequence dim, classify
        fused  = mri_tok.squeeze(1)       # (B, fusion_dim)
        logits = self.classifier(fused)   # (B, num_classes)
        return logits


# =====================================================
# ★ FULL MODEL: CNN Encoders + Transformer Fusion
# =====================================================

class BrainTumorTransformer(nn.Module):
    """
    Full model combining:
      - EfficientNet-B0 CNN (MRI encoder)    — kept from original
      - FMRIEncoder MLP (fMRI encoder)       — kept from original
      - TransformerFusion (multi-layer cross-attention) — NEW

    OLD BrainTumorCNN:
        mri (1280) + fmri (64) → cat(1344) → Linear → Linear → output

    NEW BrainTumorTransformer:
        mri (1280) → project(256) ─┐
                                    CrossAttn×3 → LayerNorm → Linear → output
        fmri (64)  → project(256) ─┘
    """
    def __init__(self, num_classes: int = 3, fmri_in_dim: int = 128,
                 dropout: float = 0.5):
        super().__init__()
        self.mri_enc  = MRIEncoder(dropout=dropout)
        self.fmri_enc = FMRIEncoder(in_dim=fmri_in_dim)
        self.fusion   = TransformerFusion(
            mri_dim     = 1280,
            fmri_dim    = 64,
            fusion_dim  = CFG["fusion_dim"],
            num_heads   = CFG["num_heads"],
            num_layers  = CFG["num_layers"],
            ffn_dim     = CFG["ffn_dim"],
            num_classes = num_classes,
            dropout     = CFG["attn_dropout"],
        )

    def forward(self, img, fmri=None):
        mri_feat = self.mri_enc(img)                         # (B, 1280)
        if fmri is None:
            fmri = torch.zeros(img.size(0), 128, device=img.device)
        fmri_feat = self.fmri_enc(fmri)                      # (B, 64)
        return self.fusion(mri_feat, fmri_feat)              # (B, num_classes)


# Build model
model    = BrainTumorTransformer(
    num_classes  = NUM_CLASSES,
    fmri_in_dim  = CFG["fmri_dim"],
    dropout      = CFG["dropout"],
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[Model] BrainTumorTransformer | use_fmri={USE_FMRI} | params={n_params:,}")
print(f"[Model] Fusion: {CFG['num_layers']} cross-attention layers, "
      f"{CFG['num_heads']} heads, dim={CFG['fusion_dim']}")


# =====================================================
# LOSS / OPTIMIZER / SCHEDULER  (unchanged)
# =====================================================

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class_wt  = torch.tensor([1.4, 1.2, 1.0], device=DEVICE)
criterion = FocalLoss(weight=class_wt, gamma=1.5)

optimizer = torch.optim.AdamW(model.parameters(),
                               lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)


# =====================================================
# TRAINING LOOP  (unchanged except model name in prints)
# =====================================================

best_acc, no_improve = 0.0, 0
train_losses, val_accs = [], []

print("\n" + "=" * 60)
print(f"  Training BrainTumorTransformer ({CFG['epochs']} epochs)")
print("=" * 60)

for epoch in range(1, CFG["epochs"] + 1):
    model.train()
    run_loss, correct, total = 0.0, 0, 0

    bar = tqdm(train_loader,
               desc=f"Ep {epoch:02d}/{CFG['epochs']}", leave=False, ncols=80)
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
            print(f"\n[OOM] Reduce batch_size (currently {CFG['batch_size']}). Try 8.\n")
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
# EVALUATION  (unchanged)
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
print(classification_report(y_true, y_pred,
                             target_names=TUMOR_CLASSES, zero_division=0))

cm  = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=TUMOR_CLASSES, yticklabels=TUMOR_CLASSES, ax=axes[0])
axes[0].set_title("Confusion Matrix (Transformer Fusion)")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
axes[0].tick_params(axis="x", rotation=30)

ep = range(1, len(train_losses) + 1)
axes[1].plot(ep, train_losses, "#e74c3c", linewidth=2, label="Train Loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss", color="#e74c3c")
ax2 = axes[1].twinx()
ax2.plot(ep, val_accs, "#3498db", linewidth=2, label="Val Acc %")
ax2.set_ylabel("Accuracy %", color="#3498db")
axes[1].set_title("Learning Curves — Transformer Fusion")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("tumor_transformer_results.png", dpi=150)
plt.show()


# =====================================================
# GRAD-CAM  (unchanged — hooks on CNN backbone)
# =====================================================

class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = [None]
        self.activations = [None]
        last_conv = model.mri_enc.features[-1]
        last_conv.register_forward_hook(
            lambda m, i, o: self.activations.__setitem__(0, o.detach()))
        last_conv.register_full_backward_hook(
            lambda m, gi, go: self.gradients.__setitem__(0, go[0].detach()))

    def generate(self, img_tensor, fmri_tensor=None, class_idx=None):
        self.model.zero_grad()
        logits = self.model(img_tensor, fmri_tensor)
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
    h, w  = orig_rgb.shape[:2]
    cam_r = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_CUBIC)
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
    fmri_dummy = torch.zeros(1, CFG["fmri_dim"], device=DEVICE)
    cam, _ = cam_engine.generate(inp, fmri_dummy, class_idx=cls_idx)
    overlay, heatmap_v, contour_img = gradcam_3panel(orig_rgb, cam)
    for r, (panel, title) in enumerate(zip(
            [img_np, overlay, contour_img], row_titles)):
        axes[r, cls_idx].imshow(panel, cmap="gray" if r == 0 else None)
        axes[r, cls_idx].set_title(
            f"{title}\n{TUMOR_CLASSES[cls_idx]}", fontsize=10)
        axes[r, cls_idx].axis("off")

plt.suptitle("Grad-CAM — BrainTumorTransformer", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("tumor_transformer_gradcam.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n[Done]")
print(f"  Best checkpoint : {BEST_MODEL}")
print(f"  Results plot    : tumor_transformer_results.png")
print(f"  Grad-CAM plot   : tumor_transformer_gradcam.png")
print(f"\n  Copy to NeuroApp:")
print(f"  copy {BEST_MODEL} neuroapp\\checkpoints\\tumor_transformer_best.pt")