from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from small_3d_resnet import build_model


BASE = Path(__file__).resolve().parent
FULL_CKPT = BASE / "full_model.ckpt"
MASKED_CKPT = BASE / "masked_model.ckpt"

TARGET_SHAPE = (96, 112, 96)
TARGET_SCALE = 50.0
EPS = 1e-8

MODEL_CONFIG: dict[str, Any] = {
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


def _load_net(ckpt_path: Path, device: str) -> torch.nn.Module:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    net = build_model(MODEL_CONFIG)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    raw_state = ckpt.get("state_dict", ckpt)
    state = {
        (key[len("model."):] if key.startswith("model.") else key): value
        for key, value in raw_state.items()
    }
    net.load_state_dict(state, strict=True)
    net.eval()
    net.to(device)
    return net


def _nifti_to_tensor(nifti_path: str | Path) -> torch.Tensor:
    image = nib.load(str(nifti_path))
    data = image.get_fdata(dtype=np.float32)

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D MRI volume, got shape {data.shape} for {nifti_path}")

    valid_mask = np.isfinite(data) & (data != 0)
    if not valid_mask.any():
        raise ValueError(f"No valid voxels found in {nifti_path}")

    valid_values = data[valid_mask].astype(np.float64, copy=False)
    mean = float(valid_values.mean())
    std = float(valid_values.std())
    scale = std if std > EPS else 1.0

    volume = ((data - mean) / scale).astype(np.float32)
    volume[~valid_mask] = 0.0
    mask = valid_mask.astype(np.float32)

    stacked = np.stack([volume, mask], axis=0).astype(np.float32, copy=False)
    stacked = _pad_or_crop(stacked, TARGET_SHAPE)
    return torch.from_numpy(stacked).unsqueeze(0).float()


def _pad_or_crop(image: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    output = np.zeros((image.shape[0], *target_shape), dtype=np.float32)
    input_slices = []
    output_slices = []

    for current, target in zip(image.shape[1:], target_shape):
        if current >= target:
            start = (current - target) // 2
            input_slices.append(slice(start, start + target))
            output_slices.append(slice(0, target))
        else:
            start = (target - current) // 2
            input_slices.append(slice(0, current))
            output_slices.append(slice(start, start + current))

    output[(slice(None), *output_slices)] = image[(slice(None), *input_slices)]
    output[1] = (output[1] > 0.5).astype(np.float32)
    output[0, output[1] == 0] = 0.0
    return output


def _inverse_target(raw_output: torch.Tensor) -> torch.Tensor:
    return torch.sinh(raw_output) * TARGET_SCALE


class LateFusionPredictor:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.full_model = _load_net(FULL_CKPT, self.device)
        self.masked_model = _load_net(MASKED_CKPT, self.device)

    def predict_from_two_niftis(
        self,
        full_mri_path: str | Path,
        masked_mri_path: str | Path,
    ) -> dict[str, float]:
        full_tensor = _nifti_to_tensor(full_mri_path).to(self.device)
        masked_tensor = _nifti_to_tensor(masked_mri_path).to(self.device)

        with torch.no_grad():
            full_raw = self.full_model(full_tensor).squeeze(-1)
            masked_raw = self.masked_model(masked_tensor).squeeze(-1)
            full_pred = _inverse_target(full_raw)
            masked_pred = _inverse_target(masked_raw)
            late_fusion_pred = 0.5 * full_pred + 0.5 * masked_pred

        return {
            "full_prediction": float(full_pred.cpu().item()),
            "masked_prediction": float(masked_pred.cpu().item()),
            "late_fusion_prediction": float(late_fusion_pred.cpu().item()),
        }

