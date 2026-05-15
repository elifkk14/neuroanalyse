from __future__ import annotations

import os
import zipfile
from pathlib import Path


def convert_to_nifti(upload_path: str, work_dir: str) -> tuple[str, str]:
    """
    Convert uploaded file to NIfTI.
    Returns (nifti_path, input_format_label).
    """
    name = Path(upload_path).name.lower()

    if name.endswith(".nii.gz") or name.endswith(".nii"):
        return upload_path, "NIfTI"

    if name.endswith(".dcm"):
        dcm_dir = os.path.dirname(upload_path)
        return _dicom_dir_to_nifti(dcm_dir, work_dir), "DICOM"

    if name.endswith(".zip"):
        extract_dir = os.path.join(work_dir, "zip_extract")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(upload_path, "r") as zf:
            zf.extractall(extract_dir)
        return _dicom_dir_to_nifti(extract_dir, work_dir), "DICOM (series)"

    raise ValueError(f"Unsupported format: {name}")


def _dicom_dir_to_nifti(dcm_dir: str, work_dir: str) -> str:
    import dicom2nifti

    out_dir = os.path.join(work_dir, "nifti_out")
    os.makedirs(out_dir, exist_ok=True)
    dicom2nifti.convert_directory(
        dcm_dir, out_dir, compression=True, reorient=True
    )

    candidates = (
        list(Path(out_dir).glob("*.nii.gz"))
        + list(Path(out_dir).glob("*.nii"))
    )
    if not candidates:
        raise ValueError(
            "DICOM conversion produced no NIfTI output. "
            "Ensure the uploaded file is a valid DICOM series."
        )
    return str(candidates[0])
