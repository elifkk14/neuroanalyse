"""
Centiloid predictor and MRI slice visualiser.

Preprocessing pipeline (must match training exactly):
  1. Load SPM-normalised NIfTI  (shape 91×109×91, MNI space)
  2. Apply DementiaMask_AAL3.nii  →  masked = data * mask
  3. valid_mask = finite & non-zero
  4. Per-volume z-score:  volume = (masked - mean) / std
  5. Stack channels:  [ch0=volume, ch1=valid_mask]
  6. Pad/crop to [96, 112, 96]
  7. Forward pass through model
  8. Inverse asinh:  centiloid = sinh(raw_output) * 50.0

IMPORTANT: The input MRI must already be spatially normalised to MNI space
(e.g. via SPM12 "Normalise" step) before uploading. Raw scanner DICOM/NIfTI
that has not been registered to MNI space will produce unreliable results.
"""
from __future__ import annotations

import base64
import math
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
# interface/backend/predictor.py  →  ../../  →  project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))   # allows: from models.small_3d_resnet import …

_CKPT_PATH  = _PROJECT_ROOT / "epoch=233-val_MAE_centiloid_raw=18.348.ckpt"
_MASK_PATH  = _PROJECT_ROOT / "DementiaMask_AAL3.nii"

# ── Constants (must match config_masked.py) ───────────────────────────────────
_TARGET_SHAPE = (96, 112, 96)
_TARGET_SCALE = 50.0   # asinh transform scale
_EPS          = 1e-8

_MODEL_CONFIG = {
    "name": "Small3DResNetRegressor",
    "in_channels": 2,
    "base_channels": 16,
    "stages": [16, 32, 64, 128],
    "blocks_per_stage": [1, 1, 1, 1],
    "norm": "groupnorm",
    "groupnorm_groups": 8,
    "activation": "silu",
    "dropout": 0.20,
    "head_dropout": 0.10,
    "output_dim": 1,
}

# ── Lazy singletons ───────────────────────────────────────────────────────────
_model     = None
_mask_data = None


def _get_mask() -> np.ndarray:
    global _mask_data
    if _mask_data is None:
        _mask_data = nib.load(str(_MASK_PATH)).get_fdata().astype(np.float32)
    return _mask_data


def _get_model():
    global _model
    if _model is not None:
        return _model

    import torch
    from models.small_3d_resnet import build_model

    net = build_model(_MODEL_CONFIG)

    ckpt = torch.load(str(_CKPT_PATH), map_location="cpu", weights_only=False)
    raw_state = ckpt.get("state_dict", ckpt)

    # Lightning saves weights as "model.<layer>…" — strip the prefix
    state = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in raw_state.items()
    }

    net.load_state_dict(state, strict=True)
    net.eval()
    _model = net
    return _model


# ── Preprocessing (mirrors dataloader.py exactly) ────────────────────────────

def _preprocess(nifti_path: str):
    import torch

    dementia_mask = _get_mask()

    img  = nib.load(nifti_path)
    data = img.get_fdata(dtype=np.float32)

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {data.shape}")

    # Apply spatial mask
    masked = data * dementia_mask

    # Valid-voxel mask: finite and non-zero
    valid = np.isfinite(masked) & (masked != 0)
    if not valid.any():
        raise ValueError("No valid voxels after masking. "
                         "Check that the MRI is in MNI space.")

    # Per-volume z-score normalisation
    vals  = masked[valid].astype(np.float64)
    mean  = float(vals.mean())
    std   = float(vals.std())
    scale = std if std > _EPS else 1.0

    volume = ((masked - mean) / scale).astype(np.float32)
    volume[~valid] = 0.0
    mask_ch = valid.astype(np.float32)

    # Stack → [2, D, H, W]  then pad/crop → [2, 96, 112, 96]
    image = _pad_or_crop(np.stack([volume, mask_ch], axis=0), _TARGET_SHAPE)

    return torch.from_numpy(image).unsqueeze(0)   # [1, 2, 96, 112, 96]


def _pad_or_crop(image: np.ndarray,
                 target: tuple[int, int, int]) -> np.ndarray:
    """Centred pad or crop each spatial dimension — identical to dataloader."""
    out = np.zeros((image.shape[0], *target), dtype=np.float32)
    in_sl, out_sl = [], []
    for cur, tgt in zip(image.shape[1:], target):
        if cur >= tgt:
            i0 = (cur - tgt) // 2
            in_sl.append(slice(i0, i0 + tgt))
            out_sl.append(slice(0, tgt))
        else:
            o0 = (tgt - cur) // 2
            in_sl.append(slice(0, cur))
            out_sl.append(slice(o0, o0 + cur))
    out[(slice(None), *out_sl)] = image[(slice(None), *in_sl)]
    out[1] = (out[1] > 0.5).astype(np.float32)
    out[0, out[1] == 0] = 0.0
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def predict_centiloid(nifti_path: str) -> tuple[float, dict]:
    import torch

    model  = _get_model()
    tensor = _preprocess(nifti_path)

    with torch.no_grad():
        raw = model(tensor).item()

    centiloid = round(math.sinh(raw) * _TARGET_SCALE, 1)

    return centiloid, {
        "name": _CKPT_PATH.name,
        "architecture": "3D ResNet Regressor",
        "status": "active",
        "val_mae_centiloid": 18.3,
    }


def generate_slices(nifti_path: str) -> dict[str, str]:
    img  = nib.load(nifti_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    x, y, z = data.shape
    return {
        "axial":    _to_b64(np.rot90(data[:, :, z // 2]), "Axial"),
        "coronal":  _to_b64(np.rot90(data[:, y // 2, :]), "Coronal"),
        "sagittal": _to_b64(np.rot90(data[x // 2, :, :]), "Sagittal"),
    }


def _to_b64(arr: np.ndarray, title: str) -> str:
    nonzero = arr[arr > 0]
    vmin, vmax = (
        (float(np.percentile(nonzero, 1)), float(np.percentile(nonzero, 99)))
        if nonzero.size else (0.0, 1.0)
    )
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="#0d0d0d")
    ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="bilinear", aspect="auto")
    ax.set_title(title, color="#aaaaaa", fontsize=8, pad=3)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight",
                facecolor="#0d0d0d")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
