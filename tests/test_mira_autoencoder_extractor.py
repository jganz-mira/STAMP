from pathlib import Path

import pytest
import torch

from stamp.preprocessing.extractor import mira_autoencoder as autoencoder_module


def _checkpoint(
    model: autoencoder_module._AutoencoderFeatureModel,
    *,
    patch_size: int = 64,
) -> dict[str, object]:
    return {
        "hyper_parameters": {
            "architecture": {
                "_target_": (
                    "src.models.architectures.autoencoder.mit.MitAutoencoder"
                ),
                "encoder": "mit_b2",
                "in_channels": 3,
                "conditioning_width": 128,
            }
        },
        "datamodule_hyper_parameters": {
            "patch_size": patch_size,
            "batch_size_inference": 3,
        },
        "model_contract": {
            "patch_size": patch_size,
            "conditioning": True,
            "preprocessing": {
                "normalize": True,
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
            },
            "scale_support": {
                "kind": "interval",
                "minimum": 0.2,
                "maximum": 1.25,
            },
            "features": {
                "stage": "deepest",
                "pooling": "global_average",
                "dimension": 512,
                "normalization": "none",
                "corruption": False,
            },
        },
        "state_dict": {
            **{f"model.{key}": value for key, value in model.state_dict().items()},
            "model.bottleneck.weight": torch.ones(1),
        },
    }


def test_mira_autoencoder_loads_conditioned_encoder_features(tmp_path: Path) -> None:
    encoder = autoencoder_module.smp.encoders.get_encoder(
        "mit_b2", in_channels=3, depth=5, weights=None
    )
    source = autoencoder_module._AutoencoderFeatureModel(
        encoder, input_size_px=64, input_mpp=0.5, conditioning_width=128
    ).eval()
    checkpoint_path = tmp_path / "autoencoder.ckpt"
    torch.save(_checkpoint(source), checkpoint_path)

    extractor = autoencoder_module.mira_autoencoder(
        checkpoint_path, input_mpp=0.5, batch_size=2
    )
    images = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        expected = source(images)
        actual = extractor.model(images)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 512)
    assert extractor.batch_size == 2
    assert extractor.input_size_px == 64
    assert extractor.input_mpp == 0.5
    assert extractor.autocast_dtype == torch.bfloat16
    assert extractor.identifier.startswith("mira_autoencoder-")
    assert extractor.identifier.endswith("-64px-0.5mpp-bf16")


def test_mira_autoencoder_validates_scale_and_native_size(tmp_path: Path) -> None:
    encoder = autoencoder_module.smp.encoders.get_encoder(
        "mit_b2", in_channels=3, depth=5, weights=None
    )
    source = autoencoder_module._AutoencoderFeatureModel(
        encoder, input_size_px=64, input_mpp=0.5, conditioning_width=128
    )
    checkpoint_path = tmp_path / "autoencoder.ckpt"
    torch.save(_checkpoint(source), checkpoint_path)

    with pytest.raises(ValueError, match="supports 0.2 to 1.25"):
        autoencoder_module.mira_autoencoder(checkpoint_path, input_mpp=1.5)

    extractor = autoencoder_module.mira_autoencoder(
        checkpoint_path, input_mpp=0.5
    )
    with pytest.raises(ValueError, match="expects native"):
        extractor.model(torch.randn(1, 3, 32, 64))
