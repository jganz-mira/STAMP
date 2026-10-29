from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torchvision.transforms import v2

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor
from stamp.utils.cache import file_digest

try:
    import segmentation_models_pytorch as smp
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "MIRA autoencoder dependencies not installed. "
        "Please reinstall stamp using `pip install 'stamp[mira_seg]'`"
    ) from e


class _ChannelModulation(nn.Module):
    """FiLM layer matching the resolution-conditioned training architecture."""

    def __init__(self, condition_width: int, channels: int) -> None:
        super().__init__()
        self.affine = nn.Linear(condition_width, 2 * channels)

    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        gamma, beta = self.affine(condition).to(features.dtype).chunk(2, dim=-1)
        if features.ndim == 4:
            gamma, beta = gamma[:, :, None, None], beta[:, :, None, None]
        elif features.ndim == 3:
            gamma, beta = gamma[:, None, :], beta[:, None, :]
        else:
            raise ValueError("FiLM expects BCHW features or BNC tokens")
        return features * (1 + gamma) + beta


class _AutoencoderFeatureModel(nn.Module):
    """Inference-only conditioned MiT-B2 encoder from the MIRA autoencoder."""

    def __init__(
        self,
        encoder: nn.Module,
        input_size_px: int,
        input_mpp: float,
        conditioning_width: int,
    ) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.encoder = encoder
        self.input_size_px = input_size_px
        self.input_mpp = input_mpp
        self.embedding = nn.Sequential(
            nn.Linear(1, conditioning_width),
            nn.SiLU(),
            nn.Linear(conditioning_width, conditioning_width),
        )
        self.null_embedding = nn.Parameter(torch.zeros(1, conditioning_width))
        channels = self.model.encoder.out_channels[2:]
        self.patch_modulation = nn.ModuleList(
            [_ChannelModulation(conditioning_width, channels_) for channels_ in channels]
        )
        self.stage_modulation = nn.ModuleList(
            [_ChannelModulation(conditioning_width, channels_) for channels_ in channels]
        )
        self.block_modulation = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                _ChannelModulation(conditioning_width, channels_),
                                _ChannelModulation(conditioning_width, channels_),
                            ]
                        )
                        for _ in getattr(self.model.encoder, f"block{index + 1}")
                    ]
                )
                for index, channels_ in enumerate(channels)
            ]
        )

    def _encode(self, images: Tensor, condition: Tensor) -> Tensor:
        encoder = self.model.encoder
        features = images
        for index in range(4):
            features = self.patch_modulation[index](
                getattr(encoder, f"patch_embed{index + 1}")(features), condition
            )
            for block, modulation in zip(
                getattr(encoder, f"block{index + 1}"),
                self.block_modulation[index],
                strict=True,
            ):
                batch, channels, height, width = features.shape
                tokens = features.flatten(2).transpose(1, 2)
                tokens = tokens + block.drop_path(
                    block.attn(modulation[0](block.norm1(tokens), condition), height, width)
                )
                tokens = tokens + block.drop_path(
                    block.mlp(modulation[1](block.norm2(tokens), condition), height, width)
                )
                features = tokens.transpose(1, 2).reshape(
                    batch, channels, height, width
                )
            features = self.stage_modulation[index](
                getattr(encoder, f"norm{index + 1}")(features), condition
            ).contiguous()
        return features

    def extract_features(self, images: Tensor) -> Tensor:
        expected_shape = (self.input_size_px, self.input_size_px)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("MIRA autoencoder expects BCHW RGB input")
        if tuple(images.shape[-2:]) != expected_shape:
            raise ValueError(
                f"MIRA autoencoder expects native {expected_shape} px patches, got "
                f"{tuple(images.shape[-2:])}"
            )
        resolution = torch.full(
            (len(images), 1),
            self.input_mpp,
            device=images.device,
            dtype=torch.float32,
        )
        condition = self.embedding(torch.log2(resolution))
        feature_map = self._encode(images, condition)
        return feature_map.mean(dim=(-2, -1))

    def forward(self, images: Tensor) -> Tensor:
        return self.extract_features(images)


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"MIRA autoencoder checkpoint does not contain a {name} mapping"
        )
    return cast(Mapping[str, Any], value)


def _checkpoint_metadata(
    checkpoint: Mapping[str, Any], input_mpp: float
) -> tuple[int, int, int, tuple[float, ...], tuple[float, ...]]:
    if not math.isfinite(input_mpp) or input_mpp <= 0:
        raise ValueError("MIRA autoencoder input_mpp must be finite and positive")

    hparams = _require_mapping(checkpoint.get("hyper_parameters"), "hyper_parameters")
    architecture = _require_mapping(hparams.get("architecture"), "architecture")
    expected_architecture = {"encoder": "mit_b2", "in_channels": 3}
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(
                f"MIRA autoencoder checkpoint architecture {key} must be "
                f"{expected!r}, got {architecture.get(key)!r}"
            )
    target = architecture.get("_target_")
    if not isinstance(target, str) or not target.endswith(".MitAutoencoder"):
        raise ValueError("Checkpoint is not a conditioned MIRA MiT autoencoder")
    conditioning_width = architecture.get("conditioning_width")
    if (
        not isinstance(conditioning_width, int)
        or isinstance(conditioning_width, bool)
        or conditioning_width <= 0
    ):
        raise ValueError("MIRA autoencoder conditioning_width must be positive")

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
        or patch_size % 32
    ):
        raise ValueError(
            "MIRA autoencoder checkpoint patch_size must be a positive multiple of 32"
        )
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError(
            "MIRA autoencoder checkpoint batch_size_inference must be positive"
        )

    model_contract = _require_mapping(
        checkpoint.get("model_contract"), "model_contract"
    )
    if model_contract.get("patch_size") != patch_size:
        raise ValueError("MIRA autoencoder checkpoint contains inconsistent patch sizes")
    if model_contract.get("conditioning") is not True:
        raise ValueError("MIRA autoencoder checkpoint must use resolution conditioning")
    preprocessing = _require_mapping(
        model_contract.get("preprocessing"), "model_contract.preprocessing"
    )
    mean, std = preprocessing.get("image_mean"), preprocessing.get("image_std")
    if (
        preprocessing.get("normalize") is not True
        or not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != 3
        or len(std) != 3
        or not all(isinstance(value, (int, float)) for value in [*mean, *std])
    ):
        raise ValueError(
            "MIRA autoencoder checkpoint has invalid normalization metadata"
        )
    scale_support = _require_mapping(
        model_contract.get("scale_support"), "model_contract.scale_support"
    )
    minimum, maximum = scale_support.get("minimum"), scale_support.get("maximum")
    if (
        scale_support.get("kind") != "interval"
        or not isinstance(minimum, (int, float))
        or not isinstance(maximum, (int, float))
        or not 0 < float(minimum) <= float(maximum)
    ):
        raise ValueError("MIRA autoencoder checkpoint has invalid scale support")
    if not float(minimum) <= input_mpp <= float(maximum):
        raise ValueError(
            f"MIRA autoencoder supports {float(minimum):g} to "
            f"{float(maximum):g} microns per pixel, got {input_mpp:g}"
        )

    features = _require_mapping(
        model_contract.get("features"), "model_contract.features"
    )
    expected_features = {
        "stage": "deepest",
        "pooling": "global_average",
        "dimension": 512,
        "normalization": "none",
        "corruption": False,
    }
    for key, expected in expected_features.items():
        if features.get(key) != expected:
            raise ValueError(
                f"MIRA autoencoder feature contract {key} must be {expected!r}, "
                f"got {features.get(key)!r}"
            )
    return (
        patch_size,
        batch_size,
        conditioning_width,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in std),
    )


def _load_feature_weights(
    model: _AutoencoderFeatureModel, checkpoint: Mapping[str, Any]
) -> None:
    state_dict = _require_mapping(checkpoint.get("state_dict"), "state_dict")
    expected = model.state_dict()
    feature_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if isinstance(key, str)
        and key.startswith("model.")
        and key.removeprefix("model.") in expected
        and isinstance(value, Tensor)
    }
    missing = sorted(set(expected) - set(feature_state))
    if missing:
        raise ValueError(
            "MIRA autoencoder checkpoint is missing feature weights: "
            + ", ".join(missing[:5])
        )
    nn.Module.load_state_dict(model, feature_state, strict=True)


def mira_autoencoder(
    checkpoint_path: Path,
    input_mpp: float,
    batch_size: int | None = None,
) -> Extractor[_AutoencoderFeatureModel]:
    """Load clean, globally pooled encoder features from a MIRA autoencoder."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"MIRA autoencoder checkpoint not found: {checkpoint_path}"
        )
    if batch_size is not None and (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("MIRA autoencoder batch_size must be positive")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("MIRA autoencoder checkpoint root must be a mapping")
    (
        input_size_px,
        checkpoint_batch_size,
        conditioning_width,
        image_mean,
        image_std,
    ) = _checkpoint_metadata(checkpoint, input_mpp)

    encoder = smp.encoders.get_encoder(
        "mit_b2", in_channels=3, depth=5, weights=None
    )
    model = _AutoencoderFeatureModel(
        cast(nn.Module, encoder), input_size_px, input_mpp, conditioning_width
    )
    _load_feature_weights(model, checkpoint)
    model.eval()
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=image_mean, std=image_std),
        ]
    )
    digest = file_digest(checkpoint_path)[:8]
    return Extractor(
        model=model,
        transform=transform,
        identifier=(
            f"{ExtractorName.MIRA_AUTOENCODER}-{digest}-{input_size_px}px-"
            f"{input_mpp:g}mpp-bf16"
        ),
        batch_size=batch_size or checkpoint_batch_size,
        input_size_px=input_size_px,
        input_mpp=input_mpp,
        autocast_dtype=torch.bfloat16,
    )
