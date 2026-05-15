from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        StochasticWeightAveraging,
    )
    from lightning.pytorch.loggers import CSVLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        StochasticWeightAveraging,
    )
    from pytorch_lightning.loggers import CSVLogger

from augmentation import build_mri_augmentation
from config_full import (
    AUGMENTATION_CONFIG,
    DATA_CONFIG,
    EXPERIMENT_CONFIG,
    METRIC_CONFIG,
    MODEL_CONFIG,
    TRAIN_CONFIG,
    VALIDATION_CONFIG,
)
from dataloader import MRIDataLoader, MRIRecord
from small_3d_resnet import build_model
from validation import clean_metric_dict, regression_metrics


class MRIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_config: dict[str, Any],
        augmentation_config: dict[str, Any],
        validation_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.data_config = copy.deepcopy(data_config)
        self.augmentation_config = copy.deepcopy(augmentation_config)
        self.validation_config = copy.deepcopy(validation_config)
        self.train_records: list[MRIRecord] = []
        self.val_records: list[MRIRecord] = []
        self.train_dataset: MRIDataLoader | None = None
        self.val_dataset: MRIDataLoader | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        if bool(self.validation_config.get("run_all_folds", False)):
            raise NotImplementedError(
                "This training entry point intentionally trains one model only. "
                "Set run_all_folds=False and choose validation_config['fold_index']."
            )

        base_dataset = MRIDataLoader(
            **self._dataset_kwargs(return_tensor=False, cache_data=False)
        )

        records = base_dataset.records
        labels_raw = np.asarray([record.label for record in records], dtype=np.float64)
        labels_for_split = labels_raw

        folds = make_stratified_folds(
            labels=labels_for_split,
            bins=self.validation_config["bins"],
            n_splits=int(self.validation_config["n_splits"]),
            seed=int(TRAIN_CONFIG["seed"]),
        )

        fold_index = int(self.validation_config["fold_index"])
        val_indices = set(folds[fold_index])

        self.train_records = [
            record for index, record in enumerate(records)
            if index not in val_indices
        ]
        self.val_records = [
            record for index, record in enumerate(records)
            if index in val_indices
        ]

        train_subjects = {record.subject for record in self.train_records}
        val_subjects = {record.subject for record in self.val_records}
        overlap = train_subjects & val_subjects

        if overlap:
            examples = sorted(overlap)[:10]
            raise RuntimeError(f"Train/validation subject leakage detected. Examples: {examples}")

        print(f"Train sample count: {len(self.train_records)}")
        print(f"Validation sample count: {len(self.val_records)}")
        print(f"Validation fold index: {fold_index}")

        train_transform = build_mri_augmentation(self.augmentation_config)

        self.train_dataset = MRIDataLoader(
            **self._dataset_kwargs(return_tensor=True, records=self.train_records),
            transform=train_transform,
        )

        self.val_dataset = MRIDataLoader(
            **self._dataset_kwargs(return_tensor=True, records=self.val_records),
            transform=None,
        )

    def train_dataloader(self) -> TorchDataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must run before train_dataloader().")

        return TorchDataLoader(
            self.train_dataset,
            batch_size=int(TRAIN_CONFIG["batch_size"]),
            shuffle=True,
            num_workers=int(self.data_config["num_workers"]),
            pin_memory=bool(self.data_config["pin_memory"]),
        )

    def val_dataloader(self) -> TorchDataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule.setup() must run before val_dataloader().")

        return TorchDataLoader(
            self.val_dataset,
            batch_size=int(TRAIN_CONFIG["batch_size"]),
            shuffle=False,
            num_workers=int(self.data_config["num_workers"]),
            pin_memory=bool(self.data_config["pin_memory"]),
        )

    def _dataset_kwargs(self, **overrides: Any) -> dict[str, Any]:
        keys = {
            "image_dir",
            "labels_csv",
            "subject_col",
            "label_col",
            "normalize",
            "treat_zero_as_missing",
            "target_shape",
            "shape_mode",
            "resize_mode_image",
            "resize_mode_mask",
            "cache_data",
            "return_tensor",
            "strict",
        }
        kwargs = {key: self.data_config[key] for key in keys if key in self.data_config}
        kwargs.update(overrides)
        return kwargs


class CentiloidRegressorModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict[str, Any],
        train_config: dict[str, Any],
        validation_config: dict[str, Any],
        metric_config: dict[str, Any],
    ) -> None:
        super().__init__()

        self.model_config = copy.deepcopy(model_config)
        self.train_config = copy.deepcopy(train_config)
        self.validation_config = copy.deepcopy(validation_config)
        self.metric_config = copy.deepcopy(metric_config)

        self.model = build_model(self.model_config)
        self.loss_fn = self._build_loss()

        self.validation_predictions: list[torch.Tensor] = []
        self.validation_targets: list[torch.Tensor] = []

        if bool(self.train_config.get("save_hyperparameters", True)):
            self.save_hyperparameters(
                {
                    "model_config": to_plain_dict(self.model_config),
                    "train_config": to_plain_dict(self.train_config),
                    "validation_config": to_plain_dict(self.validation_config),
                    "metric_config": to_plain_dict(self.metric_config),
                }
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image).squeeze(-1)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        labels_raw = batch["label"].float()
        labels = labels_raw

        pred_transformed = self(batch["image"])
        target_transformed = self.transform_target(labels)

        per_sample_loss = self.loss_fn(pred_transformed, target_transformed)
        loss = per_sample_loss.mean()

        pred_raw = self.inverse_transform_target(pred_transformed.detach())

        mae_raw = torch.mean(torch.abs(pred_raw - labels))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(labels))
        self.log(
            "train_MAE_centiloid_raw",
            mae_raw,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(labels),
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_predictions = []
        self.validation_targets = []

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        labels_raw = batch["label"].float()
        labels = labels_raw

        pred_transformed = self(batch["image"])
        target_transformed = self.transform_target(labels)

        loss = self.loss_fn(pred_transformed, target_transformed).mean()

        pred_raw = self.inverse_transform_target(pred_transformed.detach())

        self.validation_predictions.append(pred_raw.detach().cpu())
        self.validation_targets.append(labels.detach().cpu())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(labels))
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.validation_predictions:
            return

        preds = torch.cat(self.validation_predictions).numpy()
        targets = torch.cat(self.validation_targets).numpy()

        metrics = regression_metrics(
            y_true=targets,
            y_pred=preds,
            bins=self.validation_config["bins"],
            auc_thresholds=self.metric_config["amyloid_positive_auc_thresholds"],
        )

        for name, value in clean_metric_dict(metrics).items():
            self.log(
                f"val_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name == "MAE_centiloid_raw",
            )

    def configure_optimizers(self) -> Any:
        optimizer_name = str(self.train_config["optimizer"]).lower()

        if optimizer_name != "adamw":
            raise ValueError(f"Unsupported optimizer: {self.train_config['optimizer']}")

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.train_config["lr"]),
            weight_decay=float(self.train_config["weight_decay"]),
        )

        scheduler_config = self.train_config["scheduler"]

        if scheduler_config.get("name") != "cosine":
            return optimizer

        warmup_epochs = int(scheduler_config.get("warmup_epochs", 0))
        max_epochs = int(scheduler_config.get("max_epochs", 1))
        cosine_epochs = max(1, max_epochs - warmup_epochs)

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
        )

        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def transform_target(self, labels: torch.Tensor) -> torch.Tensor:
        transform_name = str(self.train_config["target_transform"]).lower()
        scale = float(self.train_config["target_scale"])

        if transform_name == "asinh":
            return torch.asinh(labels / scale)

        if transform_name in {"none", "identity"}:
            return labels

        raise ValueError(f"Unsupported target transform: {self.train_config['target_transform']}")

    def inverse_transform_target(self, transformed: torch.Tensor) -> torch.Tensor:
        transform_name = str(self.train_config["target_transform"]).lower()
        scale = float(self.train_config["target_scale"])

        if transform_name == "asinh":
            return torch.sinh(transformed) * scale

        if transform_name in {"none", "identity"}:
            return transformed

        raise ValueError(f"Unsupported target transform: {self.train_config['target_transform']}")

    def _clip_targets_if_enabled(self, labels: torch.Tensor) -> torch.Tensor:
        clip_cfg = self.train_config.get("target_clip", {})

        if not bool(clip_cfg.get("enabled", False)):
            return labels

        return torch.clamp(
            labels,
            min=float(clip_cfg.get("min", -30.0)),
            max=float(clip_cfg.get("max", 130.0)),
        )

    def _clip_predictions_if_enabled(self, preds: torch.Tensor) -> torch.Tensor:
        clip_cfg = self.train_config.get("prediction_clip", {})

        if not bool(clip_cfg.get("enabled", False)):
            return preds

        return torch.clamp(
            preds,
            min=float(clip_cfg.get("min", -30.0)),
            max=float(clip_cfg.get("max", 130.0)),
        )

    def _build_loss(self) -> nn.Module:
        loss_name = str(self.train_config["loss"]).lower()

        if loss_name == "smoothl1loss":
            return nn.SmoothL1Loss(
                beta=float(self.train_config["smooth_l1_beta"]),
                reduction="none",
            )

        if loss_name == "huberloss":
            return nn.HuberLoss(
                delta=float(self.train_config["huber_delta"]),
                reduction="none",
            )

        raise ValueError(f"Unsupported loss: {self.train_config['loss']}")


def make_stratified_folds(
    labels: np.ndarray,
    bins: list[float] | tuple[float, ...],
    n_splits: int,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.float64)

    bin_indices = np.digitize(labels, np.asarray(bins, dtype=np.float64), right=False)
    folds: list[list[int]] = [[] for _ in range(n_splits)]

    for bin_id in sorted(set(bin_indices.tolist())):
        indices = np.where(bin_indices == bin_id)[0]
        rng.shuffle(indices)

        for offset, index in enumerate(indices):
            folds[offset % n_splits].append(int(index))

    for fold in folds:
        rng.shuffle(fold)

    return folds


def _clip_np_targets_if_enabled(labels: np.ndarray) -> np.ndarray:
    clip_cfg = TRAIN_CONFIG.get("target_clip", {})

    if not bool(clip_cfg.get("enabled", False)):
        return labels

    return np.clip(
        labels,
        float(clip_cfg.get("min", -30.0)),
        float(clip_cfg.get("max", 130.0)),
    )


def resolve_precision(precision: str) -> str:
    if precision != "amp_fp16_or_bf16":
        return precision

    if not torch.cuda.is_available():
        return "32-true"

    if torch.cuda.is_bf16_supported():
        return "bf16-mixed"

    return "16-mixed"


def build_callbacks(train_config: dict[str, Any]) -> list[Any]:
    checkpoint_dir = Path(train_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:03d}-{val_MAE_centiloid_raw:.3f}",
            monitor="val_MAE_centiloid_raw",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_MAE_centiloid_raw",
            mode="min",
            patience=int(train_config["regularization"]["early_stopping_patience"]),
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if bool(train_config["regularization"].get("stochastic_weight_averaging", False)):
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=float(train_config["lr"]) * 0.1
            )
        )

    return callbacks


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_plain_dict(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_plain_dict(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value


def main() -> None:
    print("RUNNING TRAIN FILE:", __file__)
    print("CHECKPOINT DIR:", TRAIN_CONFIG["checkpoint_dir"])
    print("LOG DIR:", TRAIN_CONFIG["log_dir"])

    pl.seed_everything(int(TRAIN_CONFIG["seed"]), workers=True)
    random.seed(int(TRAIN_CONFIG["seed"]))
    np.random.seed(int(TRAIN_CONFIG["seed"]))
    torch.set_float32_matmul_precision("medium")

    datamodule = MRIDataModule(DATA_CONFIG, AUGMENTATION_CONFIG, VALIDATION_CONFIG)

    module = CentiloidRegressorModule(
        MODEL_CONFIG,
        TRAIN_CONFIG,
        VALIDATION_CONFIG,
        METRIC_CONFIG,
    )

    logger = CSVLogger(
        save_dir=str(TRAIN_CONFIG["log_dir"]),
        name=EXPERIMENT_CONFIG["name"],
        version=f"fold_{VALIDATION_CONFIG['fold_index']}",
    )

    trainer = pl.Trainer(
        accelerator=TRAIN_CONFIG["accelerator"],
        devices=TRAIN_CONFIG["devices"],
        max_epochs=int(TRAIN_CONFIG["scheduler"]["max_epochs"]),
        accumulate_grad_batches=int(TRAIN_CONFIG["accumulate_grad_batches"]),
        precision=resolve_precision(str(TRAIN_CONFIG["precision"])),
        callbacks=build_callbacks(TRAIN_CONFIG),
        logger=logger,
        log_every_n_steps=int(TRAIN_CONFIG["log_every_n_steps"]),
        default_root_dir=str(TRAIN_CONFIG["default_root_dir"]),
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    main()