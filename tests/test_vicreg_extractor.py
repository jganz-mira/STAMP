from pathlib import Path

import pytest
import torch
from torch import nn

from stamp.preprocessing import extract_
from stamp.preprocessing.extractor import Extractor
from stamp.preprocessing.extractor import vicreg as vicreg_module
from stamp.types import Microns, TilePixels


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return [self.projection(images)]


def _checkpoint(encoder: nn.Module) -> dict[str, object]:
    return {
        "hyper_parameters": {
            "architecture": {
                "backbone": "mit_b2",
                "backbone_provider": "auto",
                "in_channels": 3,
            }
        },
        "datamodule_hyper_parameters": {
            "patch_size": 8,
            "batch_size_inference": 4,
        },
        "model_contract": {
            "patch_size": 8,
            "preprocessing": {
                "normalize": True,
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
            },
            "scale_support": {"kind": "discrete", "values": [0.5]},
        },
        "state_dict": {
            **{
                f"model.backbone.encoder.{key}": value
                for key, value in encoder.state_dict().items()
            },
            "model.projector.layers.0.weight": torch.ones(2, 4),
        },
    }


def test_vicreg_uses_backbone_extract_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_encoder = _FakeEncoder()
    checkpoint_path = tmp_path / "vicreg.ckpt"
    torch.save(_checkpoint(source_encoder), checkpoint_path)
    monkeypatch.setattr(
        vicreg_module.smp.encoders,
        "get_encoder",
        lambda *args, **kwargs: _FakeEncoder(),
    )

    extractor = vicreg_module.vicreg(checkpoint_path)
    images = torch.randn(2, 3, 8, 8)

    expected = source_encoder(images)[-1].mean(dim=(-2, -1))
    actual = extractor.model(images)

    assert torch.equal(actual, expected)
    assert actual.shape == (2, 4)
    assert extractor.batch_size == 4
    assert extractor.input_size_px == 8
    assert extractor.input_mpp == 0.5
    assert extractor.autocast_dtype == torch.bfloat16
    assert extractor.identifier.startswith("vicreg-")
    assert extractor.identifier.endswith("-8px-0.5mpp-bf16")

    overridden = vicreg_module.vicreg(checkpoint_path, batch_size=8)
    assert overridden.batch_size == 8

    with pytest.raises(ValueError, match="expects native"):
        extractor.model(torch.randn(2, 3, 7, 8))


def test_extractor_rejects_wrong_physical_resolution(tmp_path: Path) -> None:
    extractor = Extractor(
        model=nn.Identity(),
        transform=lambda image: torch.empty(3, 8, 8),
        identifier="scale-test",
        input_size_px=8,
        input_mpp=0.5,
    )

    with pytest.raises(ValueError, match="requires 0.5 microns per pixel"):
        extract_(
            wsi_dir=tmp_path,
            output_dir=tmp_path / "output",
            wsi_list=None,
            cache_dir=None,
            cache_tiles_ext="jpg",
            extractor=extractor,
            tile_size_px=TilePixels(8),
            tile_size_um=Microns(8.0),
            max_workers=1,
            device="cpu",
            default_slide_mpp=None,
            brightness_cutoff=240,
            canny_cutoff=0.02,
            generate_hash=False,
        )
