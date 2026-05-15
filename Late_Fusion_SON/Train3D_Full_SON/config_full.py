from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

MRI_DIR = PROJECT_ROOT / "mri"
MRI_CENTILOIDS_CSV = PROJECT_ROOT / "mri_centiloids_filtered.csv"

ANALYSIS_REPORTS_DIR = PROJECT_ROOT / "analysis_reports"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_full_filtered_huber"
LOG_DIR = PROJECT_ROOT / "logs_full_filtered_huber"

RANDOM_SEED = 42


DATA_CONFIG = {
    "image_dir": MRI_DIR,
    "labels_csv": MRI_CENTILOIDS_CSV,
    "subject_col": "subject",
    "label_col": "centiloid",
    "normalize": True,
    "treat_zero_as_missing": True,
    "return_tensor": True,
    "strict": True,
    "target_shape": [96, 112, 96],
    "shape_mode": "pad",
    "resize_mode_image": "trilinear",
    "resize_mode_mask": "nearest",
    "cache_data": True,
    "cache_warning_gb": 100.0,
    "num_workers": 4,
    "pin_memory": True,
}


MODEL_CONFIG = {
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


AUGMENTATION_CONFIG = {
    "enabled": True,
    "spatial": {
        "random_affine": {
            "enabled": True,
            "p": 0.35,
            "rotate_degrees": [-6, 6],
            "translate_voxels": [-3, 3],
            "scale": [0.96, 1.04],
            "mode_image": "bilinear",
            "mode_mask": "nearest",
            "padding_mode": "zeros",
        },
        "random_lr_flip": {
            "enabled": True,
            "p": 0.10,
        },
    },
    "intensity_channel_0_only": {
        "random_bias_field": {
            "enabled": True,
            "p": 0.15,
            "coeff_range": [0.0, 0.25],
        },
        "random_gamma": {
            "enabled": True,
            "p": 0.25,
            "gamma_range": [0.85, 1.15],
        },
        "random_intensity_scale": {
            "enabled": True,
            "p": 0.25,
            "scale_range": [0.90, 1.10],
        },
        "random_intensity_shift": {
            "enabled": True,
            "p": 0.15,
            "shift_range": [-0.10, 0.10],
        },
        "random_gaussian_noise": {
            "enabled": True,
            "p": 0.15,
            "std_range": [0.005, 0.025],
        },
        "random_gaussian_blur": {
            "enabled": True,
            "p": 0.10,
            "sigma_range": [0.3, 0.7],
        },
    },
    "disabled": {
        "elastic_deformation": True,
        "large_rotation": True,
        "cutout": True,
        "mixup": True,
        "cutmix": True,
    },
}


TRAIN_CONFIG = {
    "seed": RANDOM_SEED,
    "input_shape": [96, 112, 96],

    "batch_size": 32,
    "accumulate_grad_batches": 1,
    "effective_batch_size": 32,

    "optimizer": "AdamW",
    "lr": 2e-4,
    "weight_decay": 1e-4,

    "scheduler": {
        "name": "cosine",
        "warmup_epochs": 5,
        "max_epochs": 600,
    },

    "loss": "HuberLoss",
    "smooth_l1_beta": 0.25,
    "huber_delta": 0.5,

    "target_transform": "asinh",
    "target_scale": 50.0,

    "target_clip": {
        "enabled": False,
    },

    "prediction_clip": {
        "enabled": False,
    },

    "weighted_loss": {
        "enabled": False,
    },

    "regularization": {
        "dropout": 0.20,
        "stochastic_weight_averaging": True,
        "early_stopping_patience": 500,
    },

    "precision": "bf16-mixed",
    "accelerator": "auto",
    "devices": "auto",
    "log_every_n_steps": 10,
    "save_hyperparameters": True,

    "default_root_dir": PROJECT_ROOT,
    "checkpoint_dir": CHECKPOINT_DIR,
    "log_dir": LOG_DIR,
}


VALIDATION_CONFIG = {
    "split": "single_stratified_holdout_fold",
    "run_all_folds": False,
    "n_splits": 5,
    "fold_index": 0,
    "stratify_by": "centiloid_bins",
    "bins": [-30, 0, 20, 50, 100, 130],
}


METRIC_CONFIG = {
    "metrics": [
        "MAE_centiloid_raw",
        "RMSE_centiloid_raw",
        "Pearson_r",
        "Spearman_r",
        "R2",
        "ACC_plusminus_5",
        "ACC_plusminus_10",
        "MAE_0_to_50",
        "ACC_0_to_50_plusminus_5",
        "ACC_0_to_50_plusminus_10",
        "MAE_by_centiloid_bin",
    ],
    "amyloid_positive_auc_thresholds": [20, 25, 50],
}


EXPERIMENT_CONFIG = {
    "name": "small3dresnet_full_filtered_huber",
    "model": MODEL_CONFIG,
    "data": DATA_CONFIG,
    "augmentation": AUGMENTATION_CONFIG,
    "train": TRAIN_CONFIG,
    "validation": VALIDATION_CONFIG,
    "metrics": METRIC_CONFIG,
}