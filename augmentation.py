from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def build_mri_augmentation(config: dict[str, Any] | None) -> "MRIAugmentation | None":
    if not config or not config.get("enabled", False):
        return None
    return MRIAugmentation(config)


class MRIAugmentation:
    """Light MRI augmentation for samples shaped (2, D, H, W).

    Spatial transforms are applied to both channels. Intensity transforms are
    applied only to channel 0 and then masked so background/missing voxels stay
    at zero while channel 1 remains binary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.spatial = config.get("spatial", {})
        self.intensity = config.get("intensity_channel_0_only", {})

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = sample["image"]
        input_was_numpy = isinstance(image, np.ndarray)
        tensor = torch.from_numpy(image).float() if input_was_numpy else image.float()

        tensor = self._apply_spatial(tensor)
        tensor = self._apply_intensity(tensor)
        tensor[1] = (tensor[1] > 0.5).float()
        tensor[0] = tensor[0] * tensor[1]

        sample["image"] = tensor.numpy().astype(np.float32, copy=False) if input_was_numpy else tensor
        return sample

    def _apply_spatial(self, image: torch.Tensor) -> torch.Tensor:
        affine_config = self.spatial.get("random_affine", {})
        if _is_enabled(affine_config) and _chance(float(affine_config.get("p", 0.0))):
            image = self._random_affine(image, affine_config)

        flip_config = self.spatial.get("random_lr_flip", {})
        if _is_enabled(flip_config) and _chance(float(flip_config.get("p", 0.0))):
            image = torch.flip(image, dims=(1,))

        return image

    def _apply_intensity(self, image: torch.Tensor) -> torch.Tensor:
        channel = image[0]
        mask = image[1] > 0.5

        bias_config = self.intensity.get("random_bias_field", {})
        if _is_enabled(bias_config) and _chance(float(bias_config.get("p", 0.0))):
            channel = _random_bias_field(channel, mask, bias_config)

        gamma_config = self.intensity.get("random_gamma", {})
        if _is_enabled(gamma_config) and _chance(float(gamma_config.get("p", 0.0))):
            channel = _random_gamma(channel, mask, gamma_config)

        scale_config = self.intensity.get("random_intensity_scale", {})
        if _is_enabled(scale_config) and _chance(float(scale_config.get("p", 0.0))):
            scale = _uniform_range(scale_config.get("scale_range", [1.0, 1.0]))
            channel = torch.where(mask, channel * scale, channel)

        shift_config = self.intensity.get("random_intensity_shift", {})
        if _is_enabled(shift_config) and _chance(float(shift_config.get("p", 0.0))):
            shift = _uniform_range(shift_config.get("shift_range", [0.0, 0.0]))
            channel = torch.where(mask, channel + shift, channel)

        noise_config = self.intensity.get("random_gaussian_noise", {})
        if _is_enabled(noise_config) and _chance(float(noise_config.get("p", 0.0))):
            std = _uniform_range(noise_config.get("std_range", [0.0, 0.0]))
            channel = torch.where(mask, channel + torch.randn_like(channel) * std, channel)

        blur_config = self.intensity.get("random_gaussian_blur", {})
        if _is_enabled(blur_config) and _chance(float(blur_config.get("p", 0.0))):
            sigma = _uniform_range(blur_config.get("sigma_range", [0.3, 0.7]))
            channel = _gaussian_blur_3d(channel, sigma)

        image = image.clone()
        image[0] = torch.where(mask, channel, torch.zeros_like(channel))
        return image

    def _random_affine(self, image: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
        _, depth, height, width = image.shape
        rotate_degrees = config.get("rotate_degrees", [-6, 6])
        translate_voxels = config.get("translate_voxels", [-3, 3])
        scale_range = config.get("scale", [0.96, 1.04])

        angles = [_uniform_range(rotate_degrees) * math.pi / 180.0 for _ in range(3)]
        scale = _uniform_range(scale_range)
        translate = [_uniform_range(translate_voxels) for _ in range(3)]

        rotation = _rotation_matrix_3d(*angles, device=image.device, dtype=image.dtype)
        matrix = rotation / scale
        theta = torch.zeros((1, 3, 4), device=image.device, dtype=image.dtype)
        theta[0, :, :3] = matrix
        theta[0, 0, 3] = 2.0 * translate[2] / max(width - 1, 1)
        theta[0, 1, 3] = 2.0 * translate[1] / max(height - 1, 1)
        theta[0, 2, 3] = 2.0 * translate[0] / max(depth - 1, 1)

        batch = image.unsqueeze(0)
        grid = F.affine_grid(theta, size=batch.shape, align_corners=False)

        image_mode = config.get("mode_image", "bilinear")
        image_mode = "bilinear" if image_mode == "trilinear" else image_mode
        padding_mode = config.get("padding_mode", "zeros")

        channel_0 = F.grid_sample(
            batch[:, 0:1],
            grid,
            mode=image_mode,
            padding_mode=padding_mode,
            align_corners=False,
        )
        channel_1 = F.grid_sample(
            batch[:, 1:2],
            grid,
            mode=config.get("mode_mask", "nearest"),
            padding_mode=padding_mode,
            align_corners=False,
        )
        output = torch.cat([channel_0, (channel_1 > 0.5).float()], dim=1).squeeze(0)
        output[0] = output[0] * output[1]
        return output


def _is_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("enabled", True))


def _chance(probability: float) -> bool:
    return random.random() < probability


def _uniform_range(values: Any) -> float:
    low, high = values
    return random.uniform(float(low), float(high))


def _rotation_matrix_3d(
    angle_x: float,
    angle_y: float,
    angle_z: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    cy, sy = math.cos(angle_y), math.sin(angle_y)
    cz, sz = math.cos(angle_z), math.sin(angle_z)

    rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        device=device,
        dtype=dtype,
    )
    ry = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        device=device,
        dtype=dtype,
    )
    rz = torch.tensor(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return rz @ ry @ rx


def _random_bias_field(channel: torch.Tensor, mask: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    coeff = _uniform_range(config.get("coeff_range", [0.0, 0.25]))
    if coeff == 0:
        return channel

    low_res = torch.randn((1, 1, 4, 4, 4), device=channel.device, dtype=channel.dtype) * coeff
    field = F.interpolate(
        low_res,
        size=channel.shape,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    field = torch.exp(field - field.mean())
    return torch.where(mask, channel * field, channel)


def _random_gamma(channel: torch.Tensor, mask: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    if not bool(mask.any()):
        return channel

    gamma = _uniform_range(config.get("gamma_range", [1.0, 1.0]))
    values = channel[mask]
    min_value = values.min()
    max_value = values.max()
    value_range = max_value - min_value
    if float(value_range.abs()) < 1e-8:
        return channel

    normalized = torch.clamp((channel - min_value) / value_range, 0.0, 1.0)
    adjusted = torch.pow(normalized, gamma) * value_range + min_value
    return torch.where(mask, adjusted, channel)


def _gaussian_blur_3d(channel: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(sigma * 3)))
    coords = torch.arange(-radius, radius + 1, device=channel.device, dtype=channel.dtype)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_3d = (
        kernel_1d[:, None, None]
        * kernel_1d[None, :, None]
        * kernel_1d[None, None, :]
    )
    kernel_3d = kernel_3d.view(1, 1, *kernel_3d.shape)
    volume = channel.unsqueeze(0).unsqueeze(0)
    blurred = F.conv3d(volume, kernel_3d, padding=radius)
    return blurred.squeeze(0).squeeze(0)
