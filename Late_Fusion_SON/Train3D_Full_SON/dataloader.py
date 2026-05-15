from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch is optional for metadata inspection.
    torch = None
    F = None
    Dataset = object


@dataclass(frozen=True)
class MRIRecord:
    subject: str
    image_path: Path
    label: float


class MRIDataLoader(Dataset):
    """Dataset-style MRI loader for centiloid regression.

    Each item returns a dictionary with:
    - image: shape (2, depth, height, width)
      - channel 0: normalized MRI volume
      - channel 1: binary validity mask, 1 where the source voxel was real data
    - label: centiloid value
    - subject: source subject/image id
    - image_path: local NIfTI path

    Missing voxels are defined as non-finite values plus zero-valued voxels by
    default. Set treat_zero_as_missing=False if zero intensity should be treated
    as real signal.

    Set target_shape=(96, 112, 96) and shape_mode="pad" for the first planned
    experiment. shape_mode="resize" is available for models that require exact
    interpolation instead of centered padding/cropping.
    """

    def __init__(
        self,
        image_dir: str | Path = "mri",
        labels_csv: str | Path = "mri_centiloids.csv",
        subject_col: str = "subject",
        label_col: str = "centiloid",
        normalize: bool = True,
        treat_zero_as_missing: bool = True,
        target_shape: Sequence[int] | None = None,
        shape_mode: str = "none",
        resize_mode_image: str = "trilinear",
        resize_mode_mask: str = "nearest",
        cache_data: bool = False,
        cache_warning_gb: float | None = None,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        return_tensor: bool = False,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        strict: bool = True,
        eps: float = 1e-8,
        records: Sequence[MRIRecord] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.labels_csv = Path(labels_csv)
        self.subject_col = subject_col
        self.label_col = label_col
        self.normalize = normalize
        self.treat_zero_as_missing = treat_zero_as_missing
        self.target_shape = tuple(int(dim) for dim in target_shape) if target_shape else None
        self.shape_mode = shape_mode.lower()
        self.resize_mode_image = resize_mode_image
        self.resize_mode_mask = resize_mode_mask
        self.cache_data = cache_data
        self.cache_warning_gb = cache_warning_gb
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.return_tensor = return_tensor
        self.transform = transform
        self.strict = strict
        self.eps = eps

        if self.return_tensor and torch is None:
            raise ImportError("return_tensor=True requires PyTorch to be installed.")
        if self.shape_mode == "resize" and torch is None:
            raise ImportError('shape_mode="resize" requires PyTorch to be installed.')
        if self.target_shape is not None and len(self.target_shape) != 3:
            raise ValueError(f"target_shape must have 3 values, got {self.target_shape}")
        if self.shape_mode not in {"none", "pad", "resize"}:
            raise ValueError('shape_mode must be one of: "none", "pad", "resize"')

        self.records = list(records) if records is not None else self._build_records()
        self._cache: list[dict[str, Any]] | None = None
        if self.cache_data:
            self._cache = [
                self._load_record(record)
                for record in tqdm(self.records, desc="Caching MRI volumes", unit="file")
            ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._cache is None:
            sample = self._load_record(self.records[index])
        else:
            sample = self._copy_cached_sample(self._cache[index])

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_tensor:
            sample["image"] = torch.from_numpy(np.asarray(sample["image"], dtype=np.float32))
            sample["label"] = torch.tensor(float(sample["label"]), dtype=torch.float32)

        return sample

    def _load_record(self, record: MRIRecord) -> dict[str, Any]:
        volume, mask, stats = self._load_volume(record.image_path)
        image = np.stack([volume, mask], axis=0).astype(np.float32, copy=False)
        image, shape_stats = self._apply_shape_mode(image)
        stats.update(shape_stats)

        return {
            "image": image,
            "label": np.float32(record.label),
            "subject": record.subject,
            "image_path": str(record.image_path),
            "stats": stats,
        }

    @staticmethod
    def _copy_cached_sample(sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": sample["image"].copy(),
            "label": np.float32(sample["label"]),
            "subject": sample["subject"],
            "image_path": sample["image_path"],
            "stats": dict(sample["stats"]),
        }

    def _build_records(self) -> list[MRIRecord]:
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Labels CSV not found: {self.labels_csv}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"MRI image directory not found: {self.image_dir}")

        labels = pd.read_csv(self.labels_csv)
        missing_columns = {self.subject_col, self.label_col} - set(labels.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in {self.labels_csv}: {sorted(missing_columns)}")

        labels[self.label_col] = pd.to_numeric(labels[self.label_col], errors="coerce")
        bad_labels = labels[labels[self.label_col].isna()]
        if len(bad_labels) and self.strict:
            examples = bad_labels[self.subject_col].astype(str).head(10).tolist()
            raise ValueError(f"Found {len(bad_labels)} rows with missing labels. Examples: {examples}")
        labels = labels.dropna(subset=[self.label_col])

        duplicate_rows = labels[labels[self.subject_col].astype(str).duplicated()]
        if len(duplicate_rows) and self.strict:
            examples = duplicate_rows[self.subject_col].astype(str).head(10).tolist()
            raise ValueError(f"Found duplicate subjects in labels CSV. Examples: {examples}")

        records: list[MRIRecord] = []
        missing_files: list[str] = []
        for row in labels.itertuples(index=False):
            subject = str(getattr(row, self.subject_col))
            image_path = self._find_image_path(subject)
            if image_path is None:
                missing_files.append(subject)
                continue
            records.append(
                MRIRecord(
                    subject=subject,
                    image_path=image_path,
                    label=float(getattr(row, self.label_col)),
                )
            )

        if missing_files and self.strict:
            examples = missing_files[:10]
            raise FileNotFoundError(
                f"Missing {len(missing_files)} MRI files referenced by {self.labels_csv}. "
                f"Examples: {examples}"
            )

        return records

    def _find_image_path(self, subject: str) -> Path | None:
        candidates = (
            self.image_dir / f"{subject}.nii",
            self.image_dir / f"{subject}.nii.gz",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_volume(self, image_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        image = nib.load(str(image_path))
        data = image.get_fdata(dtype=np.float32)

        if data.ndim == 4 and data.shape[-1] == 1:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D MRI volume, got shape {data.shape} for {image_path}")

        valid_mask = np.isfinite(data)
        if self.treat_zero_as_missing:
            valid_mask &= data != 0

        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            raise ValueError(f"No valid voxels found in {image_path}")

        valid_values = data[valid_mask].astype(np.float64, copy=False)
        mean = float(valid_values.mean())
        std = float(valid_values.std())
        scale = std if std > self.eps else 1.0

        if self.normalize:
            volume = (data - mean) / scale
            fill_value = 0.0
        else:
            volume = data.copy()
            fill_value = mean

        volume = np.asarray(volume, dtype=np.float32)
        volume[~valid_mask] = np.float32(fill_value)
        mask = valid_mask.astype(np.float32)

        stats = {
            "source_shape": tuple(int(dim) for dim in data.shape),
            "valid_voxels": valid_count,
            "missing_voxels": int(data.size - valid_count),
            "missing_fraction": float((data.size - valid_count) / data.size),
            "source_mean": mean,
            "source_std": std,
            "fill_value": fill_value,
            "normalized": self.normalize,
        }

        return volume, mask, stats

    def _apply_shape_mode(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        source_shape = tuple(int(dim) for dim in image.shape[1:])
        stats: dict[str, Any] = {
            "shape_mode": self.shape_mode,
            "pre_shape": source_shape,
            "target_shape": self.target_shape,
            "post_shape": source_shape,
        }

        if self.target_shape is None or self.shape_mode == "none":
            return image, stats

        if self.shape_mode == "pad":
            transformed = self._pad_or_crop(image, self.target_shape)
        elif self.shape_mode == "resize":
            transformed = self._resize_image_and_mask(image, self.target_shape)
        else:  # pragma: no cover - validated in __init__.
            raise ValueError(f"Unsupported shape_mode: {self.shape_mode}")

        stats["post_shape"] = tuple(int(dim) for dim in transformed.shape[1:])
        return transformed.astype(np.float32, copy=False), stats

    @staticmethod
    def _pad_or_crop(image: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
        output = np.zeros((image.shape[0], *target_shape), dtype=np.float32)
        input_slices = []
        output_slices = []

        for current, target in zip(image.shape[1:], target_shape):
            if current >= target:
                input_start = (current - target) // 2
                input_end = input_start + target
                output_start = 0
                output_end = target
            else:
                input_start = 0
                input_end = current
                output_start = (target - current) // 2
                output_end = output_start + current

            input_slices.append(slice(input_start, input_end))
            output_slices.append(slice(output_start, output_end))

        output[(slice(None), *output_slices)] = image[(slice(None), *input_slices)]
        output[1] = (output[1] > 0.5).astype(np.float32)
        output[0, output[1] == 0] = 0.0
        return output

    def _resize_image_and_mask(self, image: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
        if torch is None or F is None:
            raise ImportError('shape_mode="resize" requires PyTorch to be installed.')

        image_channel = torch.from_numpy(image[0:1]).unsqueeze(0).float()
        mask_channel = torch.from_numpy(image[1:2]).unsqueeze(0).float()
        image_mode = "trilinear" if self.resize_mode_image == "bilinear" else self.resize_mode_image

        resized_image = F.interpolate(
            image_channel,
            size=target_shape,
            mode=image_mode,
            align_corners=False if image_mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
        )
        resized_mask = F.interpolate(
            mask_channel,
            size=target_shape,
            mode=self.resize_mode_mask,
        )

        output = torch.cat([resized_image, (resized_mask > 0.5).float()], dim=1).squeeze(0)
        output[0][output[1] == 0] = 0.0
        return output.numpy().astype(np.float32, copy=False)