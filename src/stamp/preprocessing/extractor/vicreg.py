from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torchvision.transforms import v2

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor
from stamp.utils.cache import file_digest

try:
    import segmentation_models_pytorch as smp
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "VICReg dependencies not installed. "
        "Please reinstall stamp using `pip install 'stamp[mira_seg]'`"
    ) from e


_ENCODER_PREFIX = "model.backbone.encoder."


class _SMPGlobalPoolBackbone(nn.Module):
    """SMP encoder with the pooling contract used by the VICReg training model."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images: Tensor) -> Tensor:
        features = self.encoder(images)[-1]
        return self.pool(features).flatten(1)


class _VICRegFeatureModel(nn.Module):
    """Inference-only equivalent of VICRegModel without its projector."""

    def __init__(self, backbone: nn.Module, input_size_px: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_size_px = input_size_px

    def extract_features(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    def forward(
        self, images: Float[Tensor, "batch channel height width"]
    ) -> Float[Tensor, "batch feature"]:
        expected_shape = (self.input_size_px, self.input_size_px)
        if tuple(images.shape[-2:]) != expected_shape:
            raise ValueError(
                f"VICReg expects native {expected_shape} px patches, got "
                f"{tuple(images.shape[-2:])}"
            )
        return self.extract_features(images)


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"VICReg checkpoint does not contain a {name} mapping")
    return cast(Mapping[str, Any], value)


def _checkpoint_metadata(
    checkpoint: Mapping[str, Any],
) -> tuple[int, int, float, tuple[float, ...], tuple[float, ...]]:
    hparams = _require_mapping(checkpoint.get("hyper_parameters"), "hyper_parameters")
    architecture = _require_mapping(hparams.get("architecture"), "architecture")
    expected_architecture = {
        "backbone": "mit_b2",
        "in_channels": 3,
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(
                f"VICReg checkpoint architecture {key} must be {expected!r}, "
                f"got {architecture.get(key)!r}"
            )
    if architecture.get("backbone_provider") not in {"auto", "smp"}:
        raise ValueError("VICReg mit_b2 checkpoint must use the SMP backbone")

    datamodule = _require_mapping(
        checkpoint.get("datamodule_hyper_parameters"),
        "datamodule_hyper_parameters",
    )
    patch_size = datamodule.get("patch_size")
    batch_size = datamodule.get("batch_size_inference")
    if (
        not isinstance(patch_size, int)
        or isinstance(patch_size, bool)
        or patch_size <= 0
    ):
        raise ValueError("VICReg checkpoint patch_size must be a positive integer")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError(
            "VICReg checkpoint batch_size_inference must be a positive integer"
        )

    model_contract = _require_mapping(
        checkpoint.get("model_contract"), "model_contract"
    )
    if model_contract.get("patch_size") != patch_size:
        raise ValueError("VICReg checkpoint contains inconsistent patch sizes")
    preprocessing = _require_mapping(
        model_contract.get("preprocessing"), "model_contract.preprocessing"
    )
    mean = preprocessing.get("image_mean")
    std = preprocessing.get("image_std")
    if (
        preprocessing.get("normalize") is not True
        or not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != 3
        or len(std) != 3
        or not all(isinstance(value, (int, float)) for value in [*mean, *std])
    ):
        raise ValueError("VICReg checkpoint has invalid normalization metadata")
    scale_support = _require_mapping(
        model_contract.get("scale_support"), "model_contract.scale_support"
    )
    scales = scale_support.get("values")
    if (
        scale_support.get("kind") != "discrete"
        or not isinstance(scales, list)
        or len(scales) != 1
        or not isinstance(scales[0], (int, float))
    ):
        raise ValueError("VICReg checkpoint must declare one discrete input scale")
    return (
        patch_size,
        batch_size,
        float(scales[0]),
        tuple(float(value) for value in mean),
        tuple(float(value) for value in std),
    )


def _encoder_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Tensor]:
    state_dict = _require_mapping(checkpoint.get("state_dict"), "state_dict")
    encoder_state = {
        key.removeprefix(_ENCODER_PREFIX): value
        for key, value in state_dict.items()
        if isinstance(key, str)
        and key.startswith(_ENCODER_PREFIX)
        and isinstance(value, Tensor)
    }
    if not encoder_state:
        raise ValueError(
            f"VICReg checkpoint contains no encoder weights with prefix "
            f"{_ENCODER_PREFIX!r}"
        )
    return encoder_state


def vicreg(
    checkpoint_path: Path, batch_size: int | None = None
) -> Extractor[_VICRegFeatureModel]:
    """Load the VICReg backbone and expose ``model.extract_features(images)``."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VICReg checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("VICReg checkpoint root must be a mapping")

    input_size_px, checkpoint_batch_size, input_mpp, image_mean, image_std = (
        _checkpoint_metadata(checkpoint)
    )
    inference_batch_size = batch_size or checkpoint_batch_size
    encoder = smp.encoders.get_encoder("mit_b2", in_channels=3, depth=5, weights=None)
    # SMP overrides load_state_dict without the standard strict argument.
    nn.Module.load_state_dict(encoder, _encoder_state_dict(checkpoint), strict=True)

    model = _VICRegFeatureModel(
        _SMPGlobalPoolBackbone(cast(nn.Module, encoder)), input_size_px
    )
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=image_mean,
                std=image_std,
            ),
        ]
    )
    digest = file_digest(checkpoint_path)[:8]
    return Extractor(
        model=model,
        transform=transform,
        identifier=(
            f"{ExtractorName.VICREG}-{digest}-{input_size_px}px-{input_mpp:g}mpp-bf16"
        ),
        batch_size=inference_batch_size,
        input_size_px=input_size_px,
        input_mpp=input_mpp,
        autocast_dtype=torch.bfloat16,
    )
