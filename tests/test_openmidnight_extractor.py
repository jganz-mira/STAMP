import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from stamp.preprocessing.config import ExtractorName


class _FakeDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 1374, 3))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.mean(dim=(-2, -1))


def test_openmidnight_download_and_preprocessing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from stamp.preprocessing.extractor import openmidnight as module

    checkpoint_path = tmp_path / "teacher_checkpoint_load.pt"
    torch.save({"pos_embed": torch.ones(1, 261, 3)}, checkpoint_path)
    download_call: dict[str, object] = {}
    hub_call: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        download_call.update(kwargs)
        return str(checkpoint_path)

    def fake_hub_load(repo: str, model: str, **kwargs: object) -> nn.Module:
        hub_call.update(repo=repo, model=model, **kwargs)
        return _FakeDino()

    monkeypatch.setattr(module, "hf_hub_download", fake_download)
    monkeypatch.setattr(module.torch.hub, "load", fake_hub_load)

    extractor = module.openmidnight()

    assert download_call == {
        "repo_id": "SophontAI/OpenMidnight",
        "filename": "teacher_checkpoint_load.pt",
        "revision": "04399fc910e839a17088da1011021f954abf8edf",
    }
    assert hub_call == {
        "repo": "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        "model": "dinov2_vitg14_reg",
        "pretrained": False,
        "trust_repo": True,
        "skip_validation": True,
    }
    assert extractor.identifier == f"{ExtractorName.OPENMIDNIGHT}-04399fc9-bf16"
    assert extractor.input_size_px is None
    assert extractor.autocast_dtype == torch.bfloat16
    assert tuple(extractor.model.pos_embed.shape) == (1, 261, 3)

    image = Image.fromarray(np.full((80, 320, 3), 255, dtype=np.uint8))
    transformed = extractor.transform(image)
    pickle.dumps(extractor.transform)
    assert tuple(transformed.shape) == (3, 224, 224)
    assert transformed[:, 0, 0].tolist() == pytest.approx(
        [
            (1.0 - 0.485) / 0.229,
            (1.0 - 0.456) / 0.224,
            (1.0 - 0.406) / 0.225,
        ]
    )
