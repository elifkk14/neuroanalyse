"""
3D Grad-CAM for Small3DResNetRegressor.

Hooks the last nn.Sequential stage (stages[-1]) to capture
activations and gradients, then produces 3-plane overlay PNGs.
"""
from __future__ import annotations

import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def generate_gradcam_slices(
    net: torch.nn.Module,
    tensor: torch.Tensor,
) -> dict[str, str]:
    """
    Run Grad-CAM on `net` (Small3DResNetRegressor) with `tensor` [1,2,D,H,W].
    Returns dict with keys 'axial', 'coronal', 'sagittal'.
    """
    net.eval()

    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []

    # Hook last stage
    target_module = net.stages[-1]

    def fwd_hook(_, __, output):
        activations.append(output.detach().clone())

    def bwd_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach().clone())

    fh = target_module.register_forward_hook(fwd_hook)
    bh = target_module.register_full_backward_hook(bwd_hook)

    try:
        t = tensor.requires_grad_(True)
        out = net(t)          # [1]
        net.zero_grad()
        out.backward(torch.ones_like(out))
    finally:
        fh.remove()
        bh.remove()

    if not activations or not gradients:
        return {}

    act = activations[0]   # [1, C, d, h, w]
    grad = gradients[0]    # [1, C, d, h, w]

    # Global average pooling over spatial dims → channel weights
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)  # [1, C, 1, 1, 1]

    cam = (weights * act).sum(dim=1, keepdim=True)      # [1, 1, d, h, w]
    cam = F.relu(cam)

    # Upsample to input spatial shape [D, H, W]
    target_shape = tensor.shape[2:]  # (D, H, W)
    cam_up = F.interpolate(
        cam, size=tuple(target_shape), mode="trilinear", align_corners=False
    )
    cam_np = cam_up.squeeze().cpu().float().numpy()

    # Normalise to [0, 1]
    cam_min = cam_np.min()
    cam_max = cam_np.max()
    if cam_max > cam_min:
        cam_np = (cam_np - cam_min) / (cam_max - cam_min)
    else:
        cam_np = np.zeros_like(cam_np)

    # MRI channel 0 for background
    mri_np = tensor[0, 0].detach().cpu().numpy()

    D, H, W = mri_np.shape
    return {
        "axial":    _overlay_to_b64(mri_np[:, :, W // 2], cam_np[:, :, W // 2]),
        "coronal":  _overlay_to_b64(mri_np[:, H // 2, :], cam_np[:, H // 2, :]),
        "sagittal": _overlay_to_b64(mri_np[D // 2, :, :], cam_np[D // 2, :, :]),
    }


def _overlay_to_b64(mri_slice: np.ndarray, cam_slice: np.ndarray) -> str:
    mri_r = np.rot90(mri_slice)
    cam_r = np.rot90(cam_slice)

    fig, ax = plt.subplots(figsize=(3, 3), facecolor="#0d0d0d")

    # MRI background (gray)
    nonz = mri_r[mri_r != 0]
    vmin = float(np.percentile(nonz, 2)) if nonz.size else 0.0
    vmax = float(np.percentile(nonz, 98)) if nonz.size else 1.0
    ax.imshow(mri_r, cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="bilinear", aspect="auto")

    # Grad-CAM heatmap overlay
    ax.imshow(cam_r, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0,
              interpolation="bilinear", aspect="auto")
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight",
                facecolor="#0d0d0d", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
