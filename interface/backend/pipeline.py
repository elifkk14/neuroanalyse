"""
MRI Processing Pipeline
=======================
Stages (matching training notebooks exactly):

  1. DICOM → NIfTI          (dicom2nifti)
  2. Reorient to canonical   (nibabel as_closest_canonical)
  3. Resize to (91,109,91)   (scipy.ndimage.zoom  — mirrors standardize_batch1.ipynb)
  4. Float32 cast            (same dtype as training data)
  5. Apply DementiaMask      (data * mask  — mirrors masked_mrı.ipynb)
  6. Late Fusion inference   (LateFusionPredictor from model_runtime)
  7. Slice generation        (axial / coronal / sagittal base64 PNGs)

The Late Fusion predictor expects two NIfTI files:
  - full_mri.nii   : step 2-4 output (pre-masking)
  - masked_mri.nii : step 5 output

Both files are written to a temp directory, inference runs, then both are deleted.
"""
from __future__ import annotations

import base64
import math
import os
import sys
import tempfile
import time
import zipfile
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MASK_PATH = _PROJECT_ROOT / "DementiaMask_AAL3.nii"
_MODEL_RUNTIME_DIR = _PROJECT_ROOT / "model_runtime"

_TARGET_SHAPE = (91, 109, 91)   # must match training standardize step

class PipelineStep(str, Enum):
    CONVERT = "convert"
    REORIENT = "reorient"
    MASK = "mask"
    INFERENCE = "inference"
    DONE = "done"


# ── Lazy singleton for Late Fusion predictor ──────────────────────────────────

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        sys.path.insert(0, str(_MODEL_RUNTIME_DIR))
        from late_fusion_runtime import LateFusionPredictor
        _predictor = LateFusionPredictor()
    return _predictor


# ── DICOM → NIfTI conversion ──────────────────────────────────────────────────

def _convert_to_nifti(upload_path: str, work_dir: str) -> tuple[str, str]:
    """Returns (nifti_path, format_label). Handles .nii, .nii.gz, .dcm, .zip."""
    name = Path(upload_path).name.lower()

    if name.endswith(".nii.gz") or name.endswith(".nii"):
        return upload_path, "NIfTI"

    if name.endswith(".dcm"):
        dcm_dir = os.path.dirname(upload_path)
        return _dcm_dir_to_nifti(dcm_dir, work_dir), "DICOM"

    if name.endswith(".zip"):
        extract_dir = os.path.join(work_dir, "zip_extract")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(upload_path, "r") as zf:
            # Validate no path traversal
            for member in zf.namelist():
                if os.path.isabs(member) or ".." in member:
                    raise ValueError("Invalid zip: path traversal detected")
            zf.extractall(extract_dir)
        # Find the deepest directory containing .dcm files
        dcm_root = _find_dcm_root(extract_dir)
        return _dcm_dir_to_nifti(dcm_root, work_dir), "DICOM (zip)"

    raise ValueError(f"Unsupported format: {name}")


def _find_dcm_root(base_dir: str) -> str:
    """Return deepest directory that contains .dcm files, or base_dir."""
    for root, dirs, files in os.walk(base_dir):
        if any(f.lower().endswith(".dcm") for f in files):
            return root
    return base_dir


def _dcm_dir_to_nifti(dcm_dir: str, work_dir: str) -> str:
    import dicom2nifti
    out_dir = os.path.join(work_dir, "nifti_out")
    os.makedirs(out_dir, exist_ok=True)
    try:
        dicom2nifti.convert_directory(dcm_dir, out_dir, compression=False, reorient=True)
    except Exception as exc:
        raise ValueError(f"DICOM conversion failed: {exc}") from exc

    candidates = (
        list(Path(out_dir).glob("*.nii.gz"))
        + list(Path(out_dir).glob("*.nii"))
    )
    if not candidates:
        raise ValueError("DICOM conversion produced no NIfTI output. "
                         "Ensure the uploaded file is a valid DICOM series.")
    return str(candidates[0])


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_nifti(nifti_path: str, work_dir: str) -> tuple[str, str]:
    """
    Returns (full_path, masked_path) written to work_dir.
    Steps: reorient → resize(91,109,91) → float32 → apply mask.
    """
    from scipy.ndimage import zoom

    img = nib.load(nifti_path)
    img_canonical = nib.as_closest_canonical(img)
    data = img_canonical.get_fdata(dtype=np.float32)

    if data.ndim == 4:
        # Take first volume for 4D
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D MRI, got shape {data.shape}")

    # Resize to target shape
    if data.shape != _TARGET_SHAPE:
        factors = tuple(t / s for t, s in zip(_TARGET_SHAPE, data.shape))
        data = zoom(data, factors, order=1).astype(np.float32)
    else:
        data = data.astype(np.float32)

    affine = img_canonical.affine

    # Save full (pre-mask) NIfTI
    full_path = os.path.join(work_dir, "full_mri.nii")
    nib.save(nib.Nifti1Image(data, affine), full_path)

    # Load mask and apply
    if not _MASK_PATH.exists():
        raise FileNotFoundError(f"Dementia mask not found: {_MASK_PATH}")
    mask_img = nib.load(str(_MASK_PATH))
    mask_data = mask_img.get_fdata().astype(np.float32)

    if mask_data.shape != _TARGET_SHAPE:
        m_factors = tuple(t / s for t, s in zip(_TARGET_SHAPE, mask_data.shape))
        mask_data = zoom(mask_data, m_factors, order=0).astype(np.float32)

    masked_data = (data * mask_data).astype(np.float32)

    masked_path = os.path.join(work_dir, "masked_mri.nii")
    nib.save(nib.Nifti1Image(masked_data, affine), masked_path)

    return full_path, masked_path


# ── Slice generation ──────────────────────────────────────────────────────────

def _arr_to_b64(arr: np.ndarray, cmap: str = "gray") -> str:
    nonzero = arr[np.isfinite(arr) & (arr != 0)]
    if nonzero.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(nonzero, 2))
        vmax = float(np.percentile(nonzero, 98))

    fig, ax = plt.subplots(figsize=(3, 3), facecolor="#0d0d0d")
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="bilinear", aspect="auto")
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight",
                facecolor="#0d0d0d", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def generate_mri_slices(nifti_path: str) -> dict[str, str]:
    img = nib.load(nifti_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    x, y, z = data.shape
    return {
        "axial":    _arr_to_b64(np.rot90(data[:, :, z // 2])),
        "coronal":  _arr_to_b64(np.rot90(data[:, y // 2, :])),
        "sagittal": _arr_to_b64(np.rot90(data[x // 2, :, :])),
    }


# ── Risk classification ───────────────────────────────────────────────────────

def classify_centiloid(score: float) -> dict:
    if score < 25:
        return {
            "risk_category": "negative",
            "risk_label_tr": "Amiloid Negatif",
            "risk_label_en": "Amyloid Negative",
            "interpretation_tr": (
                "Tahmini Centiloid değeri pozitiflik eşiğinin (25 CL) altındadır. "
                "Bu sonuç, kortikal amiloid yükünün olmadığı veya minimal düzeyde "
                "olduğuyla tutarlıdır."
            ),
            "interpretation_en": (
                "The estimated Centiloid value is below the established positivity "
                "threshold (25 CL). This result is consistent with absent or "
                "minimal cortical amyloid burden."
            ),
        }
    if score < 50:
        return {
            "risk_category": "borderline",
            "risk_label_tr": "Sınır Değer — Klinik Korelasyon Önerilir",
            "risk_label_en": "Borderline — Clinical Correlation Advised",
            "interpretation_tr": (
                "Tahmini Centiloid değeri sınır aralığına düşmektedir (25–50 CL). "
                "Klinik korelasyon ve takip görüntüleme gerekebilir."
            ),
            "interpretation_en": (
                "The estimated Centiloid value falls in the borderline range "
                "(25–50 CL). Clinical correlation and follow-up imaging may be warranted."
            ),
        }
    if score < 100:
        return {
            "risk_category": "elevated",
            "risk_label_tr": "Yüksek Amiloid Yükü",
            "risk_label_en": "Elevated Amyloid Burden",
            "interpretation_tr": (
                "Tahmini Centiloid değeri yüksek aralıktadır (50–100 CL). "
                "Bu, orta düzey kortikal amiloid birikimiyle tutarlıdır. "
                "Nöroloji uzmanına yönlendirme ve yakın takip önerilir."
            ),
            "interpretation_en": (
                "The estimated Centiloid value is in the elevated range (50–100 CL), "
                "consistent with moderate cortical amyloid deposition. "
                "Referral to neurology and close follow-up is advised."
            ),
        }
    return {
        "risk_category": "high",
        "risk_label_tr": "Çok Yüksek Amiloid Yükü",
        "risk_label_en": "High Amyloid Burden",
        "interpretation_tr": (
            "Tahmini Centiloid değeri 100 CL'yi aşmaktadır. Bu, belirgin kortikal "
            "amiloid birikimi ile tutarlıdır. Acil nöroloji konsültasyonu ve "
            "kapsamlı değerlendirme önerilir."
        ),
        "interpretation_en": (
            "The estimated Centiloid value exceeds 100 CL, consistent with "
            "substantial cortical amyloid deposition. Urgent neurology consultation "
            "and comprehensive evaluation are advised."
        ),
    }


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    upload_path: str,
    filename: str,
    progress_cb: Callable[[str, int], None] | None = None,
) -> dict:
    """
    Run the complete MRI analysis pipeline.
    Returns a dict with all result fields for storage in the Analysis record.
    progress_cb(step_name, percent) is called at each stage.
    """

    def _progress(step: str, pct: int):
        if progress_cb:
            progress_cb(step, pct)

    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="na_pipeline_") as tmp:

        # Step 1 — DICOM → NIfTI
        _progress(PipelineStep.CONVERT, 5)
        try:
            nifti_path, input_format = _convert_to_nifti(upload_path, tmp)
        except Exception as exc:
            raise RuntimeError(f"conversion_failed: {exc}") from exc

        # Step 2-5 — Reorient + Resize + Mask
        _progress(PipelineStep.REORIENT, 20)
        try:
            full_path, masked_path = _preprocess_nifti(nifti_path, tmp)
        except ValueError as exc:
            raise RuntimeError(f"preprocessing_failed: {exc}") from exc

        # Generate MRI slices from full (pre-mask) volume
        _progress(PipelineStep.MASK, 40)
        try:
            slices = generate_mri_slices(full_path)
        except Exception:
            slices = {}
        try:
            masked_slices = generate_mri_slices(masked_path)
        except Exception:
            masked_slices = {}

        # Step 6 — Late Fusion inference
        _progress(PipelineStep.INFERENCE, 55)
        try:
            predictor = _get_predictor()
            result = predictor.predict_from_two_niftis(full_path, masked_path)
        except Exception as exc:
            raise RuntimeError(f"inference_failed: {exc}") from exc

        late_fusion = round(float(result["late_fusion_prediction"]), 1)
        full_pred   = round(float(result["full_prediction"]), 1)
        masked_pred = round(float(result["masked_prediction"]), 1)

        # Confidence interval: ±|full − masked| + base uncertainty (±10 CL)
        spread = abs(full_pred - masked_pred)
        ci_half = max(10.0, spread * 1.2)
        ci_low  = round(late_fusion - ci_half, 1)
        ci_high = round(late_fusion + ci_half, 1)

        elapsed = round(time.perf_counter() - t0, 2)
        _progress(PipelineStep.DONE, 100)

    risk = classify_centiloid(late_fusion)

    return {
        "centiloid": late_fusion,
        "full_prediction": full_pred,
        "masked_prediction": masked_pred,
        "confidence_low": ci_low,
        "confidence_high": ci_high,
        "risk_category": risk["risk_category"],
        "risk_label_tr": risk["risk_label_tr"],
        "risk_label_en": risk["risk_label_en"],
        "interpretation_tr": risk["interpretation_tr"],
        "interpretation_en": risk["interpretation_en"],
        "slice_axial":    slices.get("axial"),
        "slice_coronal":  slices.get("coronal"),
        "slice_sagittal": slices.get("sagittal"),
        "masked_slice_axial":    masked_slices.get("axial"),
        "masked_slice_coronal":  masked_slices.get("coronal"),
        "masked_slice_sagittal": masked_slices.get("sagittal"),
        "model_version": "LateFusion-v1.0",
        "processing_time_s": elapsed,
        "input_format": input_format,
    }
