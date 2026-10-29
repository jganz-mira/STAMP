from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

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
        "mira_seg dependencies not installed. "
        "Please reinstall stamp using `pip install 'stamp[mira_seg]'`"
    ) from e


_DEFAULT_CHECKPOINT_PATH = Path("checkpoint/last.ckpt")
_ENCODER_PREFIX = "model._orig_mod.model.encoder."


class _MiraSegEncoder(nn.Module):
    """MIRA's MiT-B2 encoder without its U-Net decoder or segmentation head."""

    def __init__(self, encoder: nn.Module, input_size_px: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.input_size_px = input_size_px

    def forward(
        self, batch: Float[Tensor, "batch channel height width"]
    ) -> Float[Tensor, "batch feature"]:
        expected_shape = (self.input_size_px, self.input_size_px)
        if tuple(batch.shape[-2:]) != expected_shape:
            raise ValueError(
                f"MIRA expects native {expected_shape} px patches, got "
                f"{tuple(batch.shape[-2:])}"
            )
        features = self.encoder(batch)
        last_feature_map = features[-1]
        return last_feature_map.mean(dim=(-2, -1))


def _encoder_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Tensor]:
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("MIRA checkpoint does not contain a state_dict mapping")

    encoder_state = {
        key.removeprefix(_ENCODER_PREFIX): value
        for key, value in state_dict.items()
        if isinstance(key, str)
        and key.startswith(_ENCODER_PREFIX)
        and isinstance(value, Tensor)
    }
    if not encoder_state:
        raise ValueError(
            f"MIRA checkpoint contains no encoder weights with prefix {_ENCODER_PREFIX!r}"
        )
    return encoder_state


def _checkpoint_patch_size(checkpoint: Mapping[str, Any]) -> int:
    datamodule_hparams = checkpoint.get("datamodule_hyper_parameters")
    if not isinstance(datamodule_hparams, Mapping):
        raise ValueError("MIRA checkpoint does not contain datamodule_hyper_parameters")

    patch_size = datamodule_hparams.get("patch_size")
    if (
        not isinstance(patch_size, int)
        or isinstance(patch_size, bool)
        or patch_size <= 0
    ):
        raise ValueError(
            "MIRA checkpoint does not contain a positive integer patch_size"
        )
    return patch_size


def mira_seg(checkpoint_path: Path | None = None) -> Extractor[_MiraSegEncoder]:
    checkpoint_path = checkpoint_path or _DEFAULT_CHECKPOINT_PATH
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MIRA checkpoint not found: {checkpoint_path}")

    # MIRA Lightning checkpoints contain OmegaConf metadata in addition to tensors.
    # They must therefore be loaded as trusted checkpoints. Only encoder tensors are
    # retained below; decoder, segmentation-head, loss, and optimizer state are ignored.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("MIRA checkpoint root must be a mapping")

    architecture = checkpoint.get("hyper_parameters", {}).get("architecture", {})
    if architecture.get("encoder") != "mit_b2":
        raise ValueError(
            "mira_seg expects a MIRA checkpoint with the mit_b2 encoder, got "
            f"{architecture.get('encoder')!r}"
        )

    encoder = smp.encoders.get_encoder("mit_b2", in_channels=3, depth=5, weights=None)
    # SMP's MixVisionTransformerEncoder overrides load_state_dict without the
    # standard ``strict`` argument. Call the nn.Module implementation directly
    # so a mismatching MIRA architecture cannot be accepted silently.
    nn.Module.load_state_dict(encoder, _encoder_state_dict(checkpoint), strict=True)

    input_size_px = _checkpoint_patch_size(checkpoint)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    digest = file_digest(checkpoint_path)[:8]
    return Extractor(
        model=_MiraSegEncoder(cast(nn.Module, encoder), input_size_px),
        transform=transform,
        identifier=f"{ExtractorName.MIRA_SEG}-{digest}-{input_size_px}px-bf16",
        # BF16 keeps native 1024 px MiT-B2 batches small enough for four-patch
        # inference on the target 12 GiB GPU.
        batch_size=4,
        input_size_px=input_size_px,
        autocast_dtype=torch.bfloat16,
    )


def mira_imagenet() -> Extractor[_MiraSegEncoder]:
    """MiT-B2 encoder with its generic ImageNet initialization."""
    input_size_px = 1024
    encoder = smp.encoders.get_encoder(
        "mit_b2", in_channels=3, depth=5, weights="imagenet"
    )
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return Extractor(
        model=_MiraSegEncoder(cast(nn.Module, encoder), input_size_px),
        transform=transform,
        identifier=f"mira_imagenet-mit_b2-imagenet-{input_size_px}px-bf16",
        batch_size=4,
        input_size_px=input_size_px,
        autocast_dtype=torch.bfloat16,
    )
