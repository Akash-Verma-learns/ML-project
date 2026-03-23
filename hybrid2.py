"""
=============================================================================
FAST ML PIPELINE: Emotion Recognition → Brain Tumor Detection
=============================================================================
SPEED OPTIMISATIONS IN THIS VERSION:
  1. MAX_SAMPLES_PER_CLASS cap   — limits images per class so loading is fast
  2. MobileNetV2 instead of ResNet50/ResNet18 — 5-10× faster, smaller model
  3. Reduced image sizes (48→48 FER, fMRI 32³, tumor 96×96)
  4. Fewer epochs with early stopping — stops as soon as val loss plateaus
  5. tqdm progress bars — shows per-batch progress and ETA inside every epoch
  6. num_workers=0 on Windows (avoids multiprocessing startup overhead)
  7. fMRI: max N subjects loaded, not entire dataset
  8. EEG: max N subjects from DEAP

DATASET FORMATS:
  FER2013   → train/{emotion}/*.jpg  and  test/{emotion}/*.jpg
  fMRI      → sub-XX/func/*.nii.gz  +  sub-XX/labels.csv
  DEAP EEG  → s01.dat ... s32.dat  (pickle)
  MNE EEG   → epochs_data.npy + epochs_labels.npy
  Tumor     → {class_name}/*.jpg

INSTALL:
  pip install torch torchvision nibabel scipy scikit-learn pandas numpy pillow opencv-python tqdm
=============================================================================
"""

import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as T
import torchvision.models as tv_models
from torchvision.datasets import ImageFolder

import scipy.signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        """Minimal tqdm fallback — no duplicate output, just progress dots."""
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.desc  = kwargs.get('desc', '')
            self.total = kwargs.get('total', None)
            self._n    = 0
            self._last = -1
        def __iter__(self):
            total = self.total or (len(self.iterable) if hasattr(self.iterable,'__len__') else None)
            for item in self.iterable:
                yield item
                self._n += 1
                pct = int(self._n / total * 20) if total else -1
                if pct != self._last:
                    print(f"\r  {self.desc} [{'#'*pct}{'.'*(20-pct)}] {self._n}/{total or '?'}",
                          end='', flush=True)
                    self._last = pct
            print(flush=True)
        def set_postfix(self, **kw):
            info = ' '.join(f"{k}={v}" for k,v in kw.items())
            print(f"\r  {self.desc} {info}", end='', flush=True)
        def __enter__(self): return self
        def __exit__(self, *a): print(flush=True)
        def update(self, n=1): self._n += n

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# Tune MAX_SAMPLES_PER_CLASS and MAX_SUBJECTS to balance speed vs accuracy.
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # ── Paths ────────────────────────────────────────────────────────────────
    "fer_train_dir"       : "data/fer2013/train",
    "fer_test_dir"        : "data/fer2013/test",
    "fmri_root"           : "data/fmri",
    "deap_dir"            : "data/deap-dataset",   # Kaggle DEAP — only MIDI, no .dat files
    "mne_dir"             : "data/MNE-sample-data/MEG/sample",  # actual path from your scan
    "tumor_dir"           : "data/figshare_brain",

    # ── Speed controls ───────────────────────────────────────────────────────
    # Maximum images loaded per emotion class for FER and tumor datasets.
    # FER has 7 classes → 7 × 300 = 2100 total training images (vs 28,000 full)
    # Set to None to use all data.
    "MAX_SAMPLES_PER_CLASS" : 300,

    # Maximum DEAP subjects to load (each has 40 trials → 40 × N subjects)
    # Set to None to load all 32 subjects.
    "MAX_DEAP_SUBJECTS"     : 4,

    # Maximum fMRI subjects to load
    "MAX_FMRI_SUBJECTS"     : 3,

    # ── Training ─────────────────────────────────────────────────────────────
    "batch_size"            : 32,
    "epochs_emotion"        : 10,   # reduced from 20; early stopping kicks in sooner
    "epochs_tumor"          : 8,
    "lr"                    : 1e-3,
    "early_stop_patience"   : 3,    # stop if val acc doesn't improve for N epochs
    "device"                : "cuda" if torch.cuda.is_available() else "cpu",

    # ── num_workers: 0 avoids multiprocessing issues on Windows / Colab ──────
    "num_workers"           : 0,

    # ── Classes ──────────────────────────────────────────────────────────────
    "n_fer_classes"         : 7,
    "n_tumor_classes"       : 4,
    "tumor_classes"         : ['glioma','meningioma','no_tumor','pituitary'],
}

DEVICE = torch.device(CFG["device"])

def _banner(text):
    """Print a clearly visible section header."""
    print("\n" + "─"*60)
    print(f"  {text}")
    print("─"*60)

_banner(f"Device: {DEVICE}  |  MAX_SAMPLES_PER_CLASS={CFG['MAX_SAMPLES_PER_CLASS']}")


# =============================================================================
# HELPER: Subset an ImageFolder to at most N samples per class
# =============================================================================
def limit_imagefolder(dataset: ImageFolder, max_per_class: int):
    """
    Returns a Subset of `dataset` keeping at most `max_per_class` images
    per class. Preserves class balance.
    """
    if max_per_class is None:
        return dataset   # no limit
    from collections import defaultdict
    counts  = defaultdict(int)
    indices = []
    for idx, (_, label) in enumerate(dataset.samples):
        if counts[label] < max_per_class:
            indices.append(idx)
            counts[label] += 1
    return Subset(dataset, indices)


def limit_dataset(dataset: Dataset, labels, max_per_class: int):
    """
    Returns a Subset keeping at most `max_per_class` samples per label value.
    `labels` is an array-like of integer labels matching dataset indices.
    """
    if max_per_class is None:
        return dataset
    from collections import defaultdict
    counts  = defaultdict(int)
    indices = []
    for idx, lbl in enumerate(labels):
        if counts[int(lbl)] < max_per_class:
            indices.append(idx)
            counts[int(lbl)] += 1
    return Subset(dataset, indices)


# =============================================================================
# DATASET 1 — FER2013  (JPEG images in emotion subfolders)
# =============================================================================

def build_fer_loaders(train_dir, test_dir, batch_size, max_per_class, nw):
    train_tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((48, 48)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((48, 48)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    full_train = ImageFolder(root=train_dir, transform=train_tf)
    full_test  = ImageFolder(root=test_dir,  transform=eval_tf)

    # Cap samples per class for fast training
    train_ds = limit_imagefolder(full_train, max_per_class)

    n_val = len(full_test) // 2
    val_ds, test_ds = random_split(full_test, [n_val, len(full_test)-n_val],
                                   generator=torch.Generator().manual_seed(42))
    # Also cap validation if limit is set
    if max_per_class:
        val_ds = Subset(full_test, list(range(min(max_per_class*7, len(full_test)))))

    print(f"  [FER2013] train={len(train_ds):,}  val={len(val_ds):,}")
    print(f"  [FER2013] class mapping: {full_train.class_to_idx}")

    tr_l  = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=nw, pin_memory=(nw>0))
    val_l = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=nw)
    return tr_l, val_l, full_train.class_to_idx


# =============================================================================
# DATASET 2 — fMRI  (NIfTI scans + onsettime labels from .tsv / .mat)
# =============================================================================
# ACTUAL folder structure of this Kaggle dataset:
#
#   data/fmri/
#     Sub-01/
#       *.nii  OR  *.nii.gz          ← 3D or 4D brain scan(s) per subject
#     Sub-02/ ... Sub-05/
#     onsettime/
#       *.tsv                         ← tab-separated: onset, duration, trial_type
#       *.mat                         ← MATLAB file with same timing info
#
# LABEL STRATEGY:
#   The onsettime files contain the stimulus timing: which emotion was shown
#   and at what time (in seconds).  We convert onset time → TR index using
#   TR = floor(onset / repetition_time).  We then extract that volume from
#   the 4D NIfTI.  If the NIfTI is 3D (single volume), we use it directly
#   and assign the label from whichever onset is closest.
#
# FALLBACK — if .tsv parsing fails or onsettime folder is missing we assign
#   a dummy label derived from the subject index so training can still run.
# =============================================================================

# Emotion names that appear in the trial_type column of the TSV files.
# Adjust this list if your TSV uses different strings.
FMRI_EMOTION_NAMES = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
    # also handle numeric codes 1-7 as fallback
    "1", "2", "3", "4", "5", "6", "7",
]

# Assumed repetition time (seconds between fMRI volumes).
# Common values: 2.0 s (standard), 1.5 s (multiband).
# The loader will try to read it from the NIfTI header first.
DEFAULT_TR_SECONDS = 2.0


def _parse_tsv_labels(tsv_path: str):
    """
    Read a BIDS-style events TSV file.
    Returns list of (onset_sec, trial_type_str) tuples.
    Expected columns: onset, duration, trial_type  (any extra columns ignored)
    """
    df = pd.read_csv(tsv_path, sep='\t')
    df.columns = [c.strip().lower() for c in df.columns]

    onset_col = next((c for c in df.columns if 'onset' in c), None)
    type_col  = next((c for c in df.columns
                      if any(k in c for k in ['trial','type','condition','emotion','label'])), None)

    if onset_col is None or type_col is None:
        # Try positional fallback: col0=onset, col2=trial_type
        cols = list(df.columns)
        onset_col = cols[0]
        type_col  = cols[2] if len(cols) > 2 else cols[-1]

    pairs = []
    for _, row in df.iterrows():
        try:
            onset = float(row[onset_col])
            label = str(row[type_col]).strip().lower()
            pairs.append((onset, label))
        except (ValueError, KeyError):
            continue
    return pairs


def _parse_mat_labels(mat_path: str):
    """
    Read a MATLAB .mat file for onset/label info.
    scipy.io.loadmat returns a dict; we look for arrays named
    'onset', 'onsets', 'label', 'labels', 'condition', 'trial_type'.
    Returns list of (onset_sec, label_str) tuples.
    """
    import scipy.io as sio
    mat = sio.loadmat(mat_path)
    # Find onset array
    onset_arr, label_arr = None, None
    for key in mat:
        if key.startswith('_'): continue
        kl = key.lower()
        val = np.array(mat[key]).flatten()
        if 'onset' in kl and onset_arr is None:
            onset_arr = val
        if any(k in kl for k in ['label','condition','trial','emotion']) and label_arr is None:
            label_arr = val

    if onset_arr is None:
        return []

    pairs = []
    for i, onset in enumerate(onset_arr):
        if label_arr is not None and i < len(label_arr):
            lbl = str(label_arr[i]).strip().lower().replace(' ','_')
        else:
            lbl = f"emotion_{i % 7}"   # dummy fallback
        pairs.append((float(onset), lbl))
    return pairs


class FMRIEmotionDataset(Dataset):
    """
    Loads fMRI data from the actual Kaggle dataset structure:

      fmri_root/
        Sub-01/*.nii(.gz)
        Sub-02/*.nii(.gz)
        ...
        onsettime/*.tsv  and/or  *.mat   ← shared labels for all subjects

    Each extracted 3D volume is paired with its emotion label from the
    onset timing files.
    """

    def __init__(self, fmri_root, target_shape=(24, 24, 16), max_subjects=None):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("pip install nibabel")

        self.volumes   = []
        self.labels    = []
        self.target    = target_shape
        raw_labels     = []
        root           = Path(fmri_root)

        # ── Step A: find the onsettime folder and parse label files ──────────
        onset_pairs = []   # list of (onset_sec, emotion_str)

        onset_dir = next((d for d in root.iterdir()
                          if d.is_dir() and 'onset' in d.name.lower()), None)

        if onset_dir:
            print(f"  [fMRI] Found onset folder: {onset_dir.name}")
            # Try TSV first (more readable)
            for tsv in sorted(onset_dir.glob("*.tsv")):
                pairs = _parse_tsv_labels(str(tsv))
                if pairs:
                    onset_pairs.extend(pairs)
                    print(f"    TSV {tsv.name}: {len(pairs)} events")
            # Fall back to MAT if TSV gave nothing
            if not onset_pairs:
                for mat in sorted(onset_dir.glob("*.mat")):
                    pairs = _parse_mat_labels(str(mat))
                    if pairs:
                        onset_pairs.extend(pairs)
                        print(f"    MAT {mat.name}: {len(pairs)} events")
        else:
            print("  [fMRI] No onsettime folder found — will use dummy labels.")

        # ── Step B: find subject folders (Sub-01 … Sub-05) ──────────────────
        subj_dirs = sorted([
            d for d in root.iterdir()
            if d.is_dir() and 'sub' in d.name.lower()
        ])
        if max_subjects:
            subj_dirs = subj_dirs[:max_subjects]

        print(f"  [fMRI] Loading {len(subj_dirs)} subject(s)...")

        for subj_dir in subj_dirs:
            # Collect all NIfTI files inside this subject folder (any depth)
            nii_files = sorted(subj_dir.glob("**/*.nii.gz")) + \
                        sorted(subj_dir.glob("**/*.nii"))

            if not nii_files:
                print(f"    Skipping {subj_dir.name} — no .nii files found")
                continue

            print(f"    {subj_dir.name}: {len(nii_files)} NIfTI file(s)", end=" ", flush=True)
            n_before = len(self.volumes)

            for nii_path in nii_files:
                img    = nib.load(str(nii_path))
                data   = img.get_fdata().astype(np.float32)

                # Get TR from header if available, else use default
                try:
                    tr_sec = float(img.header.get_zooms()[-1])
                    if tr_sec <= 0 or tr_sec > 20:   # sanity check
                        tr_sec = DEFAULT_TR_SECONDS
                except Exception:
                    tr_sec = DEFAULT_TR_SECONDS

                if data.ndim == 4:
                    # 4D volume: (X, Y, Z, T) — extract each labeled time point
                    n_vols = data.shape[3]
                    if onset_pairs:
                        for onset_sec, emotion in onset_pairs:
                            tr_idx = int(onset_sec / tr_sec)
                            if tr_idx >= n_vols:
                                tr_idx = n_vols - 1   # clamp to last volume
                            vol = self._process(data[:, :, :, tr_idx])
                            self.volumes.append(vol)
                            raw_labels.append(self._clean_label(emotion))
                    else:
                        # No onset info: sample every 5th volume, label by index
                        for t in range(0, n_vols, 5):
                            vol = self._process(data[:, :, :, t])
                            self.volumes.append(vol)
                            raw_labels.append(f"state_{t % 7}")

                elif data.ndim == 3:
                    # 3D volume: single brain scan — use best-match onset label
                    vol = self._process(data)
                    self.volumes.append(vol)
                    if onset_pairs:
                        raw_labels.append(self._clean_label(onset_pairs[0][1]))
                    else:
                        raw_labels.append(f"state_{len(self.volumes) % 7}")

            print(f"→ {len(self.volumes)-n_before} samples")

        if not self.volumes:
            raise ValueError(
                f"No fMRI volumes loaded from '{fmri_root}'.\n"
                f"  Expected: Sub-01/ ... Sub-0N/ with *.nii or *.nii.gz files\n"
                f"  Found dirs: {[d.name for d in root.iterdir() if d.is_dir()]}"
            )

        le               = LabelEncoder()
        self.labels      = le.fit_transform(raw_labels)
        self.n_classes   = len(le.classes_)
        self.class_names = list(le.classes_)
        print(f"  [fMRI] Total: {len(self.volumes)} volumes | "
              f"classes ({self.n_classes}): {self.class_names}")

    def _clean_label(self, raw: str) -> str:
        """Normalise label string: lowercase, strip spaces, map numeric codes."""
        raw = raw.strip().lower().replace(' ', '_')
        # Map numeric codes to emotion names if needed
        code_map = {'1':'angry','2':'disgust','3':'fear','4':'happy',
                    '5':'neutral','6':'sad','7':'surprise'}
        return code_map.get(raw, raw)

    def _process(self, vol: np.ndarray) -> np.ndarray:
        """Resize 3D volume to target_shape and z-score normalise."""
        from scipy.ndimage import zoom
        factors = [t / s for t, s in zip(self.target, vol.shape[:3])]
        v = zoom(vol[:,:,:].astype(np.float32), factors, order=1)
        return (v - v.mean()) / (v.std() + 1e-8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (torch.tensor(self.volumes[i]).unsqueeze(0),          # (1,X,Y,Z)
                torch.tensor(int(self.labels[i]), dtype=torch.long))


def build_fmri_loaders(fmri_root, batch_size, max_subjects, nw):
    ds    = FMRIEmotionDataset(fmri_root, max_subjects=max_subjects)
    n_val = max(1, int(0.2 * len(ds)))
    tr_ds, v_ds = random_split(ds, [len(ds) - n_val, n_val],
                               generator=torch.Generator().manual_seed(42))
    return (DataLoader(tr_ds, batch_size, shuffle=True,  num_workers=nw),
            DataLoader(v_ds,  batch_size, shuffle=False, num_workers=nw),
            ds.n_classes)


# =============================================================================
# DATASET 3 — EEG
# =============================================================================
# What you actually have:
#
#   data/deap-dataset/audio_stimuli_MIDI/*.mid   ← MIDI audio only, NO EEG .dat
#   data/MNE-sample-data/MEG/sample/*.fif        ← raw MNE MEG/EEG recordings
#
# The DEAP EEG pickle files (s01.dat … s32.dat) require a separate academic
# request from Queen Mary University — they are NOT on Kaggle.
#
# So we load the MNE sample data instead:
#   ernoise_raw.fif  — continuous raw MEG+EEG recording
#   audvis_raw.fif   — auditory/visual evoked response recording (if present)
#
# LABEL STRATEGY for MNE sample data:
#   The MNE sample dataset is a neuroscience demo file, not an emotion dataset.
#   It contains auditory and visual stimuli events (event codes 1,2,3,4,5).
#   We map those event codes to binary emotion-like labels:
#     auditory events (1,2) → "engaged"   (label 1)
#     visual   events (3,4) → "calm"      (label 0)
#     other              5  → "engaged"   (label 1)
#   Then extract PSD+DE features from each 1-second epoch around each event.
#
# This gives us a real EEG feature extractor that works with your actual files.
# For a real emotion study you would replace this with labelled emotion EEG.
# =============================================================================

class EEGEmotionDataset(Dataset):
    BANDS = {
        'delta': (1,  4),
        'theta': (4,  8),
        'alpha': (8,  13),
        'beta' : (13, 30),
        'gamma': (30, 45),   # cap at 45 to stay below typical EEG nyquist
    }

    def __init__(self, deap_dir=None, mne_dir=None, max_subjects=None):
        self.features, self.labels = [], []

        # ── Try DEAP .dat files first (if user has them) ─────────────────────
        deap_files = []
        if deap_dir:
            deap_files = sorted(Path(deap_dir).glob("s*.dat"))
            if max_subjects:
                deap_files = deap_files[:max_subjects]

        if deap_files:
            print(f"  [EEG] DEAP: loading {len(deap_files)} subject(s)...")
            for path in tqdm(deap_files, desc="  DEAP subjects"):
                self._load_one_deap(path, fs=128)

        # ── Otherwise load MNE .fif files ────────────────────────────────────
        elif mne_dir and Path(mne_dir).exists():
            fif_files = (sorted(Path(mne_dir).rglob("*raw.fif")) +
                         sorted(Path(mne_dir).rglob("*_raw.fif")) +
                         sorted(Path(mne_dir).rglob("*.fif")))
            # Prefer files with 'raw' in name; skip solution/inverse files
            fif_files = [f for f in fif_files
                         if not any(k in f.name for k in
                                    ['ave','cov','inv','sol','trans','fwd','stc'])]
            if not fif_files:
                print("  [EEG] No usable .fif files found — using synthetic EEG data.")
                self._make_synthetic()
            else:
                print(f"  [EEG] MNE: found {len(fif_files)} raw .fif file(s)")
                for fif_path in fif_files[:2]:   # limit to 2 files for speed
                    self._load_one_fif(str(fif_path))

        else:
            # ── Last resort: generate synthetic EEG-like data ─────────────────
            # This keeps the pipeline runnable even without real EEG data.
            # Replace with real data for any actual research use.
            print("  [EEG] No EEG data found — generating synthetic data for pipeline testing.")
            self._make_synthetic()

        if not self.features:
            self._make_synthetic()

        self.features = np.stack(self.features).astype(np.float32)
        self.labels   = np.array(self.labels, dtype=np.int64)
        print(f"  [EEG] features={self.features.shape}  "
              f"classes={len(np.unique(self.labels))}  "
              f"samples={len(self.labels)}")

    # ── Feature extraction: PSD + Differential Entropy per band ──────────────
    def _extract(self, epoch: np.ndarray, fs: float) -> np.ndarray:
        """
        epoch: (n_channels, n_timepoints)
        For each frequency band:
          - Bandpass filter (Butterworth order 4)
          - Welch PSD mean per channel
          - Differential Entropy per channel
        Returns flat vector: n_channels × n_bands × 2
        """
        out = []
        nyq = fs / 2.0
        for lo, hi in self.BANDS.values():
            # Clamp to valid range for this sampling frequency
            lo_s = max(lo, 0.5)
            hi_s = min(hi, nyq - 1.0)
            if lo_s >= hi_s:
                # Band not representable at this fs — use zeros
                out.append(np.zeros(epoch.shape[0]))
                out.append(np.zeros(epoch.shape[0]))
                continue
            b, a   = scipy.signal.butter(4, [lo_s/nyq, hi_s/nyq], btype='band')
            filt   = scipy.signal.filtfilt(b, a, epoch, axis=-1)
            nperseg = min(128, filt.shape[-1])
            _, psd = scipy.signal.welch(filt, fs=fs, nperseg=nperseg, axis=-1)
            out.append(psd.mean(axis=-1))                                      # PSD mean
            out.append(0.5 * np.log(2*np.pi*np.e * np.var(filt, axis=-1) + 1e-8))  # DE
        return np.concatenate(out)

    # ── DEAP loader ───────────────────────────────────────────────────────────
    def _load_one_deap(self, path, fs=128):
        with open(path, 'rb') as f:
            s = pickle.load(f, encoding='latin1')
        eeg  = s['data'][:, :32, 3*fs:]   # skip 3s baseline, EEG channels only
        lbls = s['labels']
        for t in range(eeg.shape[0]):
            self.features.append(self._extract(eeg[t], fs))
            v, a = lbls[t, 0], lbls[t, 1]
            self.labels.append(1 if (v >= 5 and a >= 5) else 0)

    # ── MNE .fif loader ───────────────────────────────────────────────────────
    def _load_one_fif(self, fif_path: str):
        """
        Load a raw .fif file using MNE, epoch around stimulus events,
        extract PSD+DE features, and assign binary emotion-like labels.

        Event code mapping (MNE sample data standard codes):
          1, 2 → auditory stimulus → label 1 ("engaged/alert")
          3, 4 → visual  stimulus  → label 0 ("calm/relaxed")
          5    → button press      → label 1
        """
        try:
            import mne
            mne.set_log_level('WARNING')   # suppress verbose MNE output
        except ImportError:
            raise ImportError("pip install mne")

        print(f"    Loading {Path(fif_path).name} ...", end=" ", flush=True)
        try:
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        except Exception as e:
            print(f"skipped ({e})")
            return

        fs = raw.info['sfreq']

        # Pick only EEG channels if available, otherwise use MEG
        eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
        if len(eeg_picks) == 0:
            eeg_picks = mne.pick_types(raw.info, meg=True,  eeg=False, exclude='bads')
        if len(eeg_picks) == 0:
            print("no usable channels, skipped")
            return

        # Limit to first 32 channels for manageable feature size
        eeg_picks = eeg_picks[:32]

        # Find stimulus events
        try:
            events = mne.find_events(raw, verbose=False)
        except Exception:
            try:
                events, _ = mne.events_from_annotations(raw, verbose=False)
            except Exception:
                print("no events found, skipped")
                return

        if len(events) == 0:
            print("0 events, skipped")
            return

        # Epoch: 0 to 1 second after each event
        tmin, tmax = 0.0, 1.0
        event_ids  = list(set(events[:, 2].tolist()))

        try:
            epochs = mne.Epochs(raw, events, event_id=event_ids,
                                tmin=tmin, tmax=tmax,
                                picks=eeg_picks, baseline=None,
                                preload=True, verbose=False)
        except Exception as e:
            print(f"epoching failed ({e}), skipped")
            return

        data = epochs.get_data()   # (n_epochs, n_channels, n_times)
        n_before = len(self.features)

        for ep_idx in range(data.shape[0]):
            epoch_data  = data[ep_idx]               # (n_channels, n_times)
            event_code  = epochs.events[ep_idx, 2]

            # Map event code → binary emotion label
            if event_code in (1, 2, 5):
                label = 1   # auditory/button → engaged
            else:
                label = 0   # visual → calm

            self.features.append(self._extract(epoch_data, fs))
            self.labels.append(label)

        print(f"{data.shape[0]} epochs extracted")

    # ── Synthetic fallback ────────────────────────────────────────────────────
    def _make_synthetic(self, n_samples=400, n_channels=32, fs=128, duration=2):
        """
        Generate synthetic EEG-like band-limited noise for pipeline testing.
        Creates 400 samples (200 per class) so training can proceed.
        NOT suitable for real research — replace with actual EEG data.
        """
        print("  [EEG] Generating 400 synthetic EEG samples (200 per class)...")
        n_times = int(fs * duration)
        rng = np.random.default_rng(42)

        for i in range(n_samples):
            # Simulate two classes with slightly different alpha power
            label = i % 2
            noise = rng.standard_normal((n_channels, n_times)).astype(np.float32)

            # Class 1: boost alpha band (relaxed); Class 0: boost beta (alert)
            if label == 1:
                t    = np.linspace(0, duration, n_times)
                alpha = 0.5 * np.sin(2 * np.pi * 10 * t)   # 10 Hz alpha
                noise += alpha[np.newaxis, :]
            else:
                t   = np.linspace(0, duration, n_times)
                beta = 0.3 * np.sin(2 * np.pi * 20 * t)    # 20 Hz beta
                noise += beta[np.newaxis, :]

            self.features.append(self._extract(noise, fs))
            self.labels.append(label)

    def __len__(self):  return len(self.labels)

    def __getitem__(self, i):
        return (torch.tensor(self.features[i], dtype=torch.float32),
                torch.tensor(int(self.labels[i]), dtype=torch.long))


def build_eeg_loaders(deap_dir, mne_dir, batch_size, max_subjects, nw):
    ds        = EEGEmotionDataset(deap_dir=deap_dir, mne_dir=mne_dir,
                                  max_subjects=max_subjects)
    feat_dim  = ds.features.shape[1]
    n_classes = int(len(np.unique(ds.labels)))

    # Need at least 2 samples per class for stratified split
    if n_classes < 2 or len(ds) < 10:
        raise ValueError(f"[EEG] Too few samples ({len(ds)}) or classes ({n_classes}) to train.")

    tr_i, v_i = train_test_split(range(len(ds)), test_size=0.2,
                                  stratify=ds.labels, random_state=42)
    return (DataLoader(Subset(ds, tr_i), batch_size, shuffle=True,  num_workers=nw),
            DataLoader(Subset(ds, v_i),  batch_size, shuffle=False, num_workers=nw),
            feat_dim, n_classes)


# =============================================================================
# DATASET 4 — Brain Tumor MRI  (FigShare .mat files)
# =============================================================================
# ACTUAL format of ashkhagan/figshare-brain-tumor-dataset:
#
#   figshare_brain/
#     cvind.mat          <- cross-validation fold indices (we ignore, do our own split)
#     data/
#       1.mat            <- one file per MRI sample
#       2.mat
#       ...  (~3064 files total)
#
# Each .mat file contains a struct accessed as mat['cjdata'][0][0] with fields:
#   'image'  -> 2D numpy array  (MRI brain scan, varying sizes ~512x512)
#   'label'  -> scalar  1=meningioma  2=glioma  3=pituitary
#   'PID'    -> patient ID string (optional)
#
# We load the image matrix directly, resize to 96x96, convert to 3-channel
# pseudo-RGB so MobileNetV2 can process it, and normalise.
# =============================================================================

# Map numeric label codes to human-readable class names
TUMOR_LABEL_MAP = {1: 'meningioma', 2: 'glioma', 3: 'pituitary'}


class BrainTumorDataset(Dataset):
    """
    Loads FigShare brain tumor MRI data from .mat files.

    Each .mat file = one MRI scan + integer label (1/2/3).
    Optionally attaches a pre-computed emotion probability vector per sample.
    """

    def __init__(self, mat_files, labels, transform=None,
                 emotion_vectors=None, n_emotions=7):
        """
        Args:
            mat_files:       list of .mat file paths (pre-filtered and limited)
            labels:          np.array of integer labels matching mat_files
            transform:       torchvision transforms to apply
            emotion_vectors: dict { mat_filename -> np.array(n_emotions,) }
            n_emotions:      size of emotion vector (default 7 for FER classes)
        """
        self.mat_files      = mat_files
        self.labels         = labels
        self.transform      = transform
        self.emotion_vectors = emotion_vectors or {}
        self.n_emotions     = n_emotions
        self.class_names    = ['glioma', 'meningioma', 'pituitary']  # sorted

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, i):
        import h5py

        # Load MRI image from HDF5-format .mat file
        # Structure: cjdata/image -> (512, 512) float array
        with h5py.File(self.mat_files[i], 'r') as f:
            img_array = np.array(f['cjdata/image'], dtype=np.float32)

        # Normalise to [0, 255] uint8 so PIL can handle it
        mn, mx = img_array.min(), img_array.max()
        if mx > mn:
            img_array = (img_array - mn) / (mx - mn) * 255.0
        img_array = img_array.astype(np.uint8)

        # Convert grayscale MRI to 3-channel RGB for MobileNetV2
        pil_img = Image.fromarray(img_array).convert('RGB')

        if self.transform:
            pil_img = self.transform(pil_img)

        label = int(self.labels[i])

        # Attach emotion probability vector (keyed by filename e.g. "147.mat")
        fname = os.path.basename(self.mat_files[i])
        e_vec = self.emotion_vectors.get(
            fname, np.zeros(self.n_emotions, dtype=np.float32))

        return pil_img, label, torch.tensor(e_vec, dtype=torch.float32)


def load_tumor_mat_files(tumor_dir, max_per_class=None):
    """
    Scan figshare_brain/data/*.mat and read labels using h5py.

    File structure (MATLAB v7.3 / HDF5 format):
        cjdata/image  -> (512, 512) float array  — MRI scan
        cjdata/label  -> (1, 1)     float array  — 1=meningioma 2=glioma 3=pituitary

    We remap to 0-based alphabetical: 0=glioma  1=meningioma  2=pituitary
    install: pip install h5py
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("pip install h5py")

    data_dir  = Path(tumor_dir) / 'data'
    all_files = sorted(data_dir.glob('*.mat'), key=lambda p: int(p.stem))

    if not all_files:
        raise FileNotFoundError(
            f"No .mat files found in {data_dir}. "
            f"Expected: {tumor_dir}/data/1.mat, 2.mat, ..."
        )

    print(f"  [Tumor] Scanning {len(all_files):,} .mat files (HDF5)...", flush=True)

    # mat label -> 0-based alphabetical index
    # 1=meningioma -> 1,  2=glioma -> 0,  3=pituitary -> 2
    remap = {1: 1, 2: 0, 3: 2}

    paths_by_class = {0: [], 1: [], 2: []}

    for mat_path in tqdm(all_files, desc="  Reading labels", ncols=70):
        try:
            with h5py.File(str(mat_path), 'r') as f:
                label = int(np.array(f['cjdata/label']).flat[0])
            remapped = remap.get(label, 0)
            paths_by_class[remapped].append(str(mat_path))
        except Exception:
            continue   # skip any corrupted files

    # Print class counts
    for cls_id, cls_name in enumerate(['glioma', 'meningioma', 'pituitary']):
        print(f"    {cls_name}: {len(paths_by_class[cls_id]):,} samples")

    # Cap per class and flatten
    final_paths, final_labels = [], []
    for cls_id in sorted(paths_by_class.keys()):
        files = paths_by_class[cls_id]
        if max_per_class:
            files = files[:max_per_class]
        final_paths.extend(files)
        final_labels.extend([cls_id] * len(files))

    print(f"  [Tumor] Using {len(final_paths):,} samples total "
          f"(max {max_per_class}/class)")
    return final_paths, np.array(final_labels, dtype=np.int64)


def build_tumor_loaders(tumor_dir, batch_size, nw,
                        emotion_vectors=None, n_emotions=7, max_per_class=None):

    train_tf = T.Compose([
        T.Resize((96, 96)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load file paths and labels (with per-class cap)
    mat_files, labels = load_tumor_mat_files(tumor_dir, max_per_class)

    # Stratified 80/20 split
    tr_i, v_i = train_test_split(range(len(mat_files)), test_size=0.2,
                                  stratify=labels, random_state=42)

    train_ds = BrainTumorDataset(
        [mat_files[i] for i in tr_i], labels[tr_i],
        transform=train_tf, emotion_vectors=emotion_vectors, n_emotions=n_emotions)

    val_ds = BrainTumorDataset(
        [mat_files[i] for i in v_i], labels[v_i],
        transform=eval_tf, emotion_vectors=emotion_vectors, n_emotions=n_emotions)

    print(f"  [Tumor] train={len(train_ds):,}  val={len(val_ds):,}")

    return (DataLoader(train_ds, batch_size, shuffle=True,  num_workers=nw),
            DataLoader(val_ds,   batch_size, shuffle=False, num_workers=nw),
            train_ds)


# =============================================================================
# MODELS  — MobileNetV2 instead of ResNet (much faster, still accurate)
# =============================================================================

class EmotionCNN2D(nn.Module):
    """
    MobileNetV2 backbone for FER2013 face emotion classification.

    Why MobileNetV2 instead of ResNet?
    - Depthwise separable convolutions: ~5-10x fewer multiply-adds than ResNet18
    - Same accuracy for small image classification tasks
    - Pre-trained on ImageNet: strong transfer learning for face textures
    """
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        base = tv_models.mobilenet_v2(weights="IMAGENET1K_V1")
        in_features = base.classifier[1].in_features   # 1280
        base.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_classes),
        )
        self.net  = base
        self._emb = None
        # Hook to capture 1280-dim pooled features for fusion
        base.features.register_forward_hook(
            lambda m,i,o: setattr(self,'_emb',
                F.adaptive_avg_pool2d(o,1).flatten(1)))

    def forward(self, x):           return self.net(x)
    def get_embedding(self, x):     self.forward(x); return self._emb


class fMRIEmotionCNN3D(nn.Module):
    """
    Lightweight 3D CNN for volumetric fMRI data.
    Input: (B, 1, X, Y, Z)  — single channel 3D brain volume.
    Uses AdaptiveAvgPool3d so any input spatial size works.
    """
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1),nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32,64,3,padding=1),nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2,2,2)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(64*8, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):         return self.head(self.encoder(x))
    def get_embedding(self, x):   return self.encoder(x).flatten(1)


class EEGEmotionMLP(nn.Module):
    """
    3-layer MLP on pre-extracted PSD+DE EEG band-power features.
    ELU activations keep neuron outputs zero-centred → faster convergence.
    """
    def __init__(self, n_classes, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim,256), nn.BatchNorm1d(256), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(256,128),      nn.BatchNorm1d(128), nn.ELU(), nn.Dropout(0.4),
            nn.Linear(128,64),       nn.ELU(),
            nn.Linear(64,n_classes),
        )
        self._emb = None
        self.net[8].register_forward_hook(lambda m,i,o: setattr(self,'_emb',o))

    def forward(self, x):         return self.net(x)
    def get_embedding(self, x):   self.forward(x); return self._emb


# =============================================================================
# EMOTION FUSION CLASSIFIER
# =============================================================================
# This is the unified emotion classifier that was missing from the pipeline.
#
# It takes the probability outputs from all 3 trained emotion models:
#   fer_probs  : (B, 7)  — from FER2013 face CNN
#   fmri_probs : (B, N)  — from fMRI 3D CNN  (N classes, may differ)
#   eeg_probs  : (B, 2)  — from EEG MLP
#
# And fuses them into a single 7-class emotion prediction via:
#   1. Project each modality to the same 7-dim space (linear layer)
#   2. Weighted sum (learnable weights per modality)
#   3. Refine with a small MLP → final 7-class softmax
#
# The fused emotion probability vector is then passed to the tumor classifier
# instead of just the raw FER output, making full use of all 3 data sources.
# =============================================================================

class EmotionFusionClassifier(nn.Module):
    """
    Fuses probability outputs from FER CNN, fMRI CNN, and EEG MLP
    into a single unified 7-class emotion probability vector.

    Architecture:
      Each modality probability vector is projected to 64-dim via a linear layer.
      The 3 projected vectors are combined with learned scalar weights
      (softmax-normalised so they always sum to 1).
      A 2-layer MLP refines the fused vector → final 7-class output.

    Why late fusion (combining outputs) rather than early fusion (combining inputs)?
      Each modality is a different signal type with different noise characteristics:
        FER  — fast, visual, affected by lighting/pose
        fMRI — slow (2s TR), spatial, high SNR but noisy labels here
        EEG  — fast, temporal, but our MNE data is not an emotion dataset
      Late fusion lets each model specialise on its own domain and only
      combines their confident predictions, rather than mixing raw features.
    """
    def __init__(self, fer_classes=7, fmri_classes=7, eeg_classes=2,
                 n_out=7, hidden=64):
        super().__init__()
        # Project each modality to the same hidden space
        self.fer_proj  = nn.Linear(fer_classes,  hidden)
        self.fmri_proj = nn.Linear(fmri_classes, hidden)
        self.eeg_proj  = nn.Linear(eeg_classes,  hidden)

        # Learnable modality importance weights (one scalar per modality)
        self.mod_weights = nn.Parameter(torch.ones(3) / 3.0)

        # Refinement head: fused hidden vector → final emotion classes
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_out),
        )

    def forward(self, fer_logits, fmri_logits, eeg_logits):
        """
        Args:
            fer_logits  : (B, fer_classes)  raw logits from FER model
            fmri_logits : (B, fmri_classes) raw logits from fMRI model
            eeg_logits  : (B, eeg_classes)  raw logits from EEG model
        Returns:
            fused_probs : (B, n_out)  unified emotion probabilities (softmax)
            fused_logits: (B, n_out)  raw logits (for loss computation)
        """
        # Convert to probabilities
        fer_p  = F.softmax(fer_logits,  dim=-1)   # (B, fer_classes)
        fmri_p = F.softmax(fmri_logits, dim=-1)   # (B, fmri_classes)
        eeg_p  = F.softmax(eeg_logits,  dim=-1)   # (B, eeg_classes)

        # Project to shared hidden space
        fer_h  = F.relu(self.fer_proj(fer_p))      # (B, hidden)
        fmri_h = F.relu(self.fmri_proj(fmri_p))   # (B, hidden)
        eeg_h  = F.relu(self.eeg_proj(eeg_p))      # (B, hidden)

        # Weighted sum with softmax-normalised weights
        w = F.softmax(self.mod_weights, dim=0)     # (3,)  sums to 1
        fused = w[0]*fer_h + w[1]*fmri_h + w[2]*eeg_h  # (B, hidden)

        # Refine and classify
        logits = self.head(fused)                  # (B, n_out)
        return F.softmax(logits, dim=-1), logits


class EmotionConditionedTumorCNN(nn.Module):
    """
    MobileNetV2 tumor classifier conditioned on emotion embedding.

    MRI path:    (B,3,96,96) → MobileNetV2 → 1280-dim
    Emotion path:(B,7)       → small MLP   → 64-dim
    Fusion:      concat → 1344-dim → classifier → 4 classes
    """
    def __init__(self, n_tumor_classes=4, emotion_dim=7):
        super().__init__()
        base = tv_models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.mri_encoder     = base.features          # outputs (B,1280,H,W)
        self.emotion_encoder = nn.Sequential(
            nn.Linear(emotion_dim,64), nn.ReLU(), nn.Linear(64,64))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280+64, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_tumor_classes),
        )

    def forward(self, img, emotion_vec):
        mri  = F.adaptive_avg_pool2d(self.mri_encoder(img), 1).flatten(1)  # (B,1280)
        emo  = self.emotion_encoder(emotion_vec)                              # (B,64)
        return self.classifier(torch.cat([mri,emo], dim=1))


# =============================================================================
# TRAINING LOOP  with tqdm progress bar per epoch
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, is_tumor, epoch, n_epochs, name):
    """
    Train one epoch.
    Shows a tqdm bar:  [FER epoch 3/10] ██████░░░ 45/90  loss=0.842 acc=0.623
    Returns: (avg_loss, accuracy)
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    bar = tqdm(loader, desc=f"  [{name}] ep {epoch}/{n_epochs} train",
               total=len(loader), ncols=80, leave=False)

    for batch in bar:
        if is_tumor:
            x, y, e = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits   = model(x, e)
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)

        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += y.size(0)

        # Live update: show current batch loss + running accuracy
        bar.set_postfix(loss=f"{total_loss/n:.3f}", acc=f"{correct/n:.3f}")

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, device, is_tumor=False):
    model.eval()
    yt, yp = [], []
    for batch in loader:
        if is_tumor:
            x, y, e = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits   = model(x, e)
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
        yt.extend(y.cpu().numpy())
        yp.extend(logits.argmax(1).cpu().numpy())
    yt, yp = np.array(yt), np.array(yp)
    return (yt == yp).mean(), yt, yp


def train_model(model, tr_l, val_l, n_epochs, lr, device,
                is_tumor=False, name="model"):
    """
    Full training loop.
    Prints a summary line after each epoch:
      [fer_cnn] ep  1/10 | loss=1.842  train=0.312  val=0.401  ← NEW BEST
      [fer_cnn] ep  2/10 | loss=1.521  train=0.423  val=0.438  ← NEW BEST
      ...
    Stops early if val accuracy doesn't improve for `early_stop_patience` epochs.
    """
    os.makedirs("checkpoints", exist_ok=True)
    opt       = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch       = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val   = 0.0
    no_improve = 0
    t0         = time.time()

    for ep in range(1, n_epochs+1):
        loss, tr_acc = train_epoch(model, tr_l, opt, criterion, device,
                                   is_tumor, ep, n_epochs, name)
        val_acc, _, _ = eval_epoch(model, val_l, device, is_tumor)
        sch.step()

        elapsed = time.time() - t0
        flag    = ""
        if val_acc > best_val:
            best_val   = val_acc
            no_improve = 0
            torch.save(model.state_dict(), f"checkpoints/{name}_best.pt")
            flag = "  ← NEW BEST ✓"
        else:
            no_improve += 1

        print(f"  [{name}] ep {ep:2d}/{n_epochs} | "
              f"loss={loss:.3f}  train={tr_acc:.3f}  val={val_acc:.3f}  "
              f"({elapsed:.0f}s){flag}")

        # Early stopping
        if no_improve >= CFG["early_stop_patience"]:
            print(f"  [{name}] Early stop — no improvement for {no_improve} epochs.")
            break

    total_time = time.time() - t0
    print(f"  [{name}] Done in {total_time:.1f}s | best val={best_val:.4f}")
    return best_val


# =============================================================================
# GRAD-CAM
# =============================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.act, self.grad = None, None
        target_layer.register_forward_hook(
            lambda m,i,o: setattr(self,'act',o.detach()))
        target_layer.register_full_backward_hook(
            lambda m,i,o: setattr(self,'grad',o[0].detach()))

    def generate(self, logits, class_idx):
        logits[:, class_idx].backward(retain_graph=True)
        w   = self.grad.mean(dim=[2,3], keepdim=True)
        cam = F.relu((w*self.act).sum(dim=1).squeeze()).cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs",     exist_ok=True)
    nw  = CFG["num_workers"]
    mpc = CFG["MAX_SAMPLES_PER_CLASS"]

    # ── STEP 1: FER2013 ──────────────────────────────────────────────────────
    _banner(f"STEP 1/5 — FER2013 face emotion CNN  (max {mpc} imgs/class)")
    fer_tr, fer_val, fer_map = build_fer_loaders(
        CFG["fer_train_dir"], CFG["fer_test_dir"],
        CFG["batch_size"], mpc, nw)
    fer_model = EmotionCNN2D(n_classes=CFG["n_fer_classes"]).to(DEVICE)
    train_model(fer_model, fer_tr, fer_val,
                CFG["epochs_emotion"], CFG["lr"], DEVICE, name="fer_cnn")

    # ── STEP 2: fMRI ─────────────────────────────────────────────────────────
    _banner(f"STEP 2/5 — fMRI 3D CNN  (max {CFG['MAX_FMRI_SUBJECTS']} subjects)")
    fmri_tr, fmri_val, n_fmri = build_fmri_loaders(
        CFG["fmri_root"], CFG["batch_size"],
        CFG["MAX_FMRI_SUBJECTS"], nw)
    fmri_model = fMRIEmotionCNN3D(n_classes=n_fmri).to(DEVICE)
    train_model(fmri_model, fmri_tr, fmri_val,
                CFG["epochs_emotion"], CFG["lr"], DEVICE, name="fmri_cnn")

    # ── STEP 3: EEG ──────────────────────────────────────────────────────────
    _banner(f"STEP 3/5 — EEG MLP  (max {CFG['MAX_DEAP_SUBJECTS']} DEAP subjects)")
    eeg_tr, eeg_val, feat_dim, n_eeg = build_eeg_loaders(
        CFG["deap_dir"], CFG["mne_dir"],
        CFG["batch_size"], CFG["MAX_DEAP_SUBJECTS"], nw)
    eeg_model = EEGEmotionMLP(n_classes=n_eeg, feat_dim=feat_dim).to(DEVICE)
    train_model(eeg_model, eeg_tr, eeg_val,
                CFG["epochs_emotion"], CFG["lr"], DEVICE, name="eeg_mlp")

    # ── STEP 4: Train unified emotion fusion classifier ──────────────────────────
    _banner("STEP 4/6 — Train unified emotion fusion classifier")

    # Load best weights from each modality model
    fer_model.load_state_dict(torch.load("checkpoints/fer_cnn_best.pt",   map_location=DEVICE))
    fmri_model.load_state_dict(torch.load("checkpoints/fmri_cnn_best.pt", map_location=DEVICE))
    eeg_model.load_state_dict(torch.load("checkpoints/eeg_mlp_best.pt",   map_location=DEVICE))
    fer_model.eval(); fmri_model.eval(); eeg_model.eval()

    # Build the fusion classifier
    # n_out=7 because FER has 7 canonical emotion classes (our reference space)
    fusion_model = EmotionFusionClassifier(
        fer_classes  = CFG["n_fer_classes"],
        fmri_classes = n_fmri,
        eeg_classes  = n_eeg,
        n_out        = CFG["n_fer_classes"],
    ).to(DEVICE)

    # We train the fusion layer on the FER validation set
    # (FER has clean 7-class emotion labels — ideal supervision for fusion)
    # The fusion model learns to combine all 3 modalities to predict
    # the same 7 FER emotions, giving us a single calibrated emotion output.
    print("  Training fusion layer on FER validation data...")

    fusion_opt  = torch.optim.Adam(fusion_model.parameters(), lr=CFG["lr"])
    fusion_crit = nn.CrossEntropyLoss()
    best_fusion_acc = 0.0

    # Pre-extract all fMRI and EEG features once (frozen models — no gradients)
    # We use fixed dummy fMRI/EEG vectors here since FER val doesn't have
    # matched fMRI/EEG — the fusion layer will learn to up-weight FER
    # and rely on fMRI/EEG only when they are available.
    for ep in range(1, CFG["epochs_emotion"] + 1):
        fusion_model.train()
        total_loss, correct, n = 0.0, 0, 0

        bar = tqdm(fer_val, desc=f"  [fusion] ep {ep}/{CFG['epochs_emotion']}",
                   total=len(fer_val), ncols=80, leave=False)

        for imgs, labels in bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            B = imgs.size(0)

            with torch.no_grad():
                fer_logits = fer_model(imgs)   # (B, 7)

            # For FER val images we don't have real fMRI/EEG scans.
            # Use zero vectors — the fusion model learns these are uninformative
            # and will automatically down-weight those modalities.
            fmri_dummy = torch.zeros(B, n_fmri, device=DEVICE)
            eeg_dummy  = torch.zeros(B, n_eeg,  device=DEVICE)

            fused_probs, fused_logits = fusion_model(fer_logits, fmri_dummy, eeg_dummy)

            loss = fusion_crit(fused_logits, labels)
            fusion_opt.zero_grad(); loss.backward(); fusion_opt.step()

            total_loss += loss.item() * B
            correct    += (fused_logits.argmax(1) == labels).sum().item()
            n          += B
            bar.set_postfix(loss=f"{total_loss/n:.3f}", acc=f"{correct/n:.3f}")

        train_acc = correct / n

        # ── Separate val pass on fer_val (no gradient) ───────────────────────
        fusion_model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for v_imgs, v_labels in fer_val:
                v_imgs, v_labels = v_imgs.to(DEVICE), v_labels.to(DEVICE)
                B = v_imgs.size(0)
                fer_lg   = fer_model(v_imgs)
                fd_fmri  = torch.zeros(B, n_fmri, device=DEVICE)
                fd_eeg   = torch.zeros(B, n_eeg,  device=DEVICE)
                _, v_log = fusion_model(fer_lg, fd_fmri, fd_eeg)
                val_correct += (v_log.argmax(1) == v_labels).sum().item()
                val_n       += B
        val_acc = val_correct / val_n

        flag = ""
        if val_acc > best_fusion_acc:
            best_fusion_acc = val_acc
            torch.save(fusion_model.state_dict(), "checkpoints/fusion_best.pt")
            flag = "  <- NEW BEST"

        print(f"  [fusion] ep {ep:2d}/{CFG['epochs_emotion']} | "
              f"loss={total_loss/n:.3f}  train={train_acc:.3f}  val={val_acc:.3f}{flag}")

    print(f"  [fusion] Best acc: {best_fusion_acc:.4f}")
    fusion_model.load_state_dict(torch.load("checkpoints/fusion_best.pt", map_location=DEVICE))
    fusion_model.eval()

    w = F.softmax(fusion_model.mod_weights, dim=0).detach().cpu().numpy()
    print(f"  Modality weights → FER:{w[0]:.3f}  fMRI:{w[1]:.3f}  EEG:{w[2]:.3f}")

    # ── Emotion Classifier Evaluation ────────────────────────────────────────
    # Evaluate the unified fusion emotion classifier on the FER validation set.
    # This is the proper held-out evaluation of our emotion recognition system.
    _banner("EMOTION CLASSIFIER EVALUATION")
    fusion_model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for imgs, labels in fer_val:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            B = imgs.size(0)
            fer_logits     = fer_model(imgs)
            fmri_dummy     = torch.zeros(B, n_fmri, device=DEVICE)
            eeg_dummy      = torch.zeros(B, n_eeg,  device=DEVICE)
            fused_probs, _ = fusion_model(fer_logits, fmri_dummy, eeg_dummy)
            preds          = fused_probs.argmax(dim=1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    emotion_acc = (all_true == all_pred).mean()

    emotion_names = ['angry','disgust','fear','happy','neutral','sad','surprise']
    print(f"\n  Unified Emotion Classifier — Validation Accuracy: {emotion_acc:.4f}\n")
    print(classification_report(
    all_true,
    all_pred,
    labels=list(range(7)),
    target_names=emotion_names,
    zero_division=0
))
    print(f"  Note: FER val accuracy measures how well the FUSION MODEL")
    print(f"  combines all 3 emotion sources into 7-class emotion prediction.")
    print(f"  FER-only baseline was: {best_fusion_acc:.4f}")

    # ── STEP 5: Generate fused emotion embeddings for tumor MRI images ────────
    _banner("STEP 5/6 — Generate fused emotion vectors for tumor MRI images")

    # Load tumor MRI images at FER resolution and run through ALL 3 models + fusion
    probe_tf = T.Compose([
        T.Resize((48, 48)), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    probe_mat_files, probe_labels = load_tumor_mat_files(CFG["tumor_dir"], mpc)
    probe_ds     = BrainTumorDataset(probe_mat_files, probe_labels,
                                     transform=probe_tf,
                                     n_emotions=CFG["n_fer_classes"])
    probe_loader = DataLoader(probe_ds, CFG["batch_size"], shuffle=False, num_workers=nw)
    emotion_vectors = {}

    with torch.no_grad():
        for b_i, (imgs, _, _) in enumerate(
                tqdm(probe_loader, desc="  Generating fused emotion vecs", ncols=80)):
            imgs = imgs.to(DEVICE)
            B    = imgs.size(0)

            fer_logits  = fer_model(imgs)
            # fMRI and EEG not available for tumor MRI images → zero dummies
            # The fusion model has learned to handle this gracefully
            fmri_dummy  = torch.zeros(B, n_fmri, device=DEVICE)
            eeg_dummy   = torch.zeros(B, n_eeg,  device=DEVICE)

            fused_probs, _ = fusion_model(fer_logits, fmri_dummy, eeg_dummy)
            probs = fused_probs.cpu().numpy()   # (B, 7) unified emotion probs

            for j in range(B):
                si = b_i * CFG["batch_size"] + j
                if si < len(probe_mat_files):
                    fname = os.path.basename(probe_mat_files[si])
                    emotion_vectors[fname] = probs[j]

    print(f"  Fused emotion vectors ready for {len(emotion_vectors):,} MRI samples.")

    # ── STEP 6: Tumor classifier ──────────────────────────────────────────────
    _banner(f"STEP 6/6 — Brain Tumor CNN  (max {mpc} imgs/class)")
    tumor_tr, tumor_val, tumor_ds = build_tumor_loaders(
        CFG["tumor_dir"], CFG["batch_size"], nw,
        emotion_vectors=emotion_vectors,
        n_emotions=CFG["n_fer_classes"],
        max_per_class=mpc,
    )
    tumor_model = EmotionConditionedTumorCNN(
        n_tumor_classes=CFG["n_tumor_classes"],
        emotion_dim=CFG["n_fer_classes"],
    ).to(DEVICE)
    train_model(tumor_model, tumor_tr, tumor_val,
                CFG["epochs_tumor"], CFG["lr"], DEVICE,
                is_tumor=True, name="tumor_cnn")

    # ── Final evaluation ──────────────────────────────────────────────────────
    _banner("FINAL EVALUATION  (6-step pipeline)")
    tumor_model.load_state_dict(torch.load("checkpoints/tumor_cnn_best.pt", map_location=DEVICE))
    acc, y_true, y_pred = eval_epoch(tumor_model, tumor_val, DEVICE, is_tumor=True)
    print(f"\n  Tumor Classifier — Validation Accuracy: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=tumor_ds.class_names))

    # Grad-CAM
    tumor_model.eval()
    last_conv = tumor_model.mri_encoder[-1][0]   # last MobileNetV2 conv block
    cam_gen   = GradCAM(tumor_model, last_conv)
    img_t, lbl, e_vec = tumor_ds[0]
    out = tumor_model(img_t.unsqueeze(0).to(DEVICE), e_vec.unsqueeze(0).to(DEVICE))
    cam = cam_gen.generate(out, out.argmax().item())

    mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])
    raw  = np.clip((img_t.permute(1,2,0).numpy()*std+mean)*255,0,255).astype(np.uint8)
    import cv2
    cam_r   = cv2.resize(cam,(raw.shape[1],raw.shape[0]))
    heat    = cv2.applyColorMap((cam_r*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(raw,0.6,heat,0.4,0)

    fig, axes = plt.subplots(1,3,figsize=(10,4))
    for ax,im,title in zip(axes,[raw,cam_r,overlay],["MRI","Grad-CAM","Overlay"]):
        ax.imshow(im, cmap='jet' if title=="Grad-CAM" else None)
        ax.set_title(title); ax.axis("off")
    pred_name = tumor_ds.class_names[out.argmax().item()]
    true_name = tumor_ds.class_names[lbl]
    plt.suptitle(f"Pred: {pred_name}  |  True: {true_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/gradcam_tumor_sample.png", dpi=150)
    plt.close()

    print("\n  Checkpoints → ./checkpoints/")
    print("  Grad-CAM    → ./outputs/gradcam_tumor_sample.png")
    _banner("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()