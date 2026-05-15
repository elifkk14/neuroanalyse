from __future__ import annotations

from pathlib import Path
import importlib
import sys
import types
import torch


def _ensure_torchvision_compat() -> None:
    """Allow older local Torch builds to import newer torchvision/torchmetrics stacks."""
    if not hasattr(torch, "library"):
        torch.library = types.SimpleNamespace()
    if hasattr(torch.library, "register_fake"):
        return

    def register_fake(*_args, **_kwargs):
        def decorator(fn):
            return fn
        return decorator

    torch.library.register_fake = register_fake


_ensure_torchvision_compat()


# =========================================================
# BASE PATH
# =========================================================

BASE = Path(__file__).resolve().parent

FULL_PROJECT = BASE / "Train3D_Full_SON"
MASKED_PROJECT = BASE / "Train3D_Masked_SON"


# =========================================================
# CHECKPOINTS
# =========================================================

FULL_CKPT = (
    FULL_PROJECT
    / "epoch=196-val_MAE_centiloid_raw=17.871.ckpt"
)

MASKED_CKPT = (
    MASKED_PROJECT
    / "epoch=279-val_MAE_centiloid_raw=17.183.ckpt"
)

TARGET_SHAPE = [96, 112, 96]


# =========================================================
# IMPORT RESET
# =========================================================

def _reset_imports():

    for name in [
        "config_full",
        "config_masked",
        "dataloader",
        "augmentation",
        "small_3d_resnet",
        "validation",
        "train_full",
        "train_masked",
    ]:
        sys.modules.pop(name, None)

    sys.path = [
        p for p in sys.path
        if "Train3D_Full_SON" not in str(p)
        and "Train3D_Masked_SON" not in str(p)
    ]


def _import_from(project: Path, module_name: str):

    _reset_imports()

    sys.path.insert(0, str(project))

    return importlib.import_module(module_name)


# =========================================================
# LOAD MODELS
# =========================================================

def _load_full_model(device: str):

    train_full = _import_from(
        FULL_PROJECT,
        "train_full"
    )

    model = train_full.CentiloidRegressorModule.load_from_checkpoint(
        str(FULL_CKPT),
        map_location=device,
    )

    model.eval()
    model.to(device)

    return model


def _load_masked_model(device: str):

    train_masked = _import_from(
        MASKED_PROJECT,
        "train_masked"
    )

    model = train_masked.CentiloidRegressorModule.load_from_checkpoint(
        str(MASKED_CKPT),
        map_location=device,
    )

    model.eval()
    model.to(device)

    return model


# =========================================================
# NIFTI TO TENSOR
# =========================================================

def _nifti_to_tensor(
    project: Path,
    config_name: str,
    nifti_path: str | Path,
) -> torch.Tensor:

    nifti_path = Path(nifti_path)

    if not nifti_path.exists():
        raise FileNotFoundError(
            f"NIfTI file not found: {nifti_path}"
        )

    _reset_imports()

    sys.path.insert(0, str(project))

    dataloader = importlib.import_module("dataloader")
    config = importlib.import_module(config_name)

    record = dataloader.MRIRecord(
        subject=nifti_path.stem,
        image_path=nifti_path,
        label=0.0,
    )

    dataset = dataloader.MRIDataLoader(
        image_dir=nifti_path.parent,
        labels_csv=config.DATA_CONFIG["labels_csv"],
        normalize=True,
        treat_zero_as_missing=True,
        target_shape=TARGET_SHAPE,
        shape_mode="pad",
        resize_mode_image="trilinear",
        resize_mode_mask="nearest",
        cache_data=False,
        return_tensor=True,
        strict=False,
        records=[record],
    )

    sample = dataset[0]

    tensor = sample["image"].unsqueeze(0).float()

    return tensor


# =========================================================
# MAIN PREDICTOR
# =========================================================

class LateFusionPredictor:

    def __init__(self, device: str | None = None):

        if not FULL_CKPT.exists():
            raise FileNotFoundError(
                f"Full checkpoint not found: {FULL_CKPT}"
            )

        if not MASKED_CKPT.exists():
            raise FileNotFoundError(
                f"Masked checkpoint not found: {MASKED_CKPT}"
            )

        self.device = (
            device
            or
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.full_model = _load_full_model(self.device)
        self.masked_model = _load_masked_model(self.device)

    def predict_from_two_niftis(
        self,
        full_mri_path: str | Path,
        masked_mri_path: str | Path,
    ):

        full_tensor = _nifti_to_tensor(
            FULL_PROJECT,
            "config_full",
            full_mri_path,
        ).to(self.device)

        masked_tensor = _nifti_to_tensor(
            MASKED_PROJECT,
            "config_masked",
            masked_mri_path,
        ).to(self.device)

        with torch.no_grad():

            full_out = self.full_model(full_tensor)
            masked_out = self.masked_model(masked_tensor)

            full_pred = self.full_model.inverse_transform_target(
                full_out
            )

            masked_pred = self.masked_model.inverse_transform_target(
                masked_out
            )

            late_fusion_pred = (
                0.5 * full_pred
                +
                0.5 * masked_pred
            )

        return {

            "full_prediction": float(
                full_pred.cpu().item()
            ),

            "masked_prediction": float(
                masked_pred.cpu().item()
            ),

            "late_fusion_prediction": float(
                late_fusion_pred.cpu().item()
            ),
        }


# =========================================================
# SIMPLE API
# =========================================================

_predictor = None


def predict_centiloid(
    full_mri_path,
    masked_mri_path,
):

    global _predictor

    if _predictor is None:
        _predictor = LateFusionPredictor()

    return _predictor.predict_from_two_niftis(
        full_mri_path,
        masked_mri_path,
    )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    predictor = LateFusionPredictor()

    print("Late Fusion Predictor Loaded")
    print("Device:", predictor.device)
