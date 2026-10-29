"""OpenMidnight tile feature extractor."""

from collections.abc import Callable
from typing import cast

try:
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from torch import nn
    from torchvision import transforms
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "OpenMidnight dependencies not installed. Please reinstall stamp using "
        "`pip install 'stamp[cpu]'` or `pip install 'stamp[gpu]'`."
    ) from e

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor

_HF_REPO_ID = "SophontAI/OpenMidnight"
_HF_FILENAME = "teacher_checkpoint_load.pt"
# Pin the released weights so the extractor identifier remains reproducible.
_HF_REVISION = "04399fc910e839a17088da1011021f954abf8edf"

# DINOv2 supplies the exact ViT-g/14-with-registers implementation used by the
# release. Pinning its source avoids silently changing the architecture.
_DINOV2_REPOSITORY = (
    "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)


def _to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _load_model() -> nn.Module:
    checkpoint_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_HF_FILENAME,
        revision=_HF_REVISION,
    )

    model = cast(
        nn.Module,
        torch.hub.load(
            _DINOV2_REPOSITORY,
            "dinov2_vitg14_reg",
            pretrained=False,
            trust_repo=True,
            skip_validation=True,
        ),
    )
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    # OpenMidnight was trained at 224 px, whereas the upstream DINOv2 model is
    # constructed with a 518 px positional embedding.
    model.pos_embed = nn.Parameter(state_dict["pos_embed"])
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model


def openmidnight() -> Extractor[nn.Module]:
    """Load OpenMidnight and return its 1536-D CLS-token embeddings."""

    transform = cast(
        Callable[[Image.Image], torch.Tensor],
        transforms.Compose(
            [
                transforms.Lambda(_to_rgb),
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        ),
    )

    return Extractor(
        model=_load_model(),
        transform=transform,
        identifier=f"{ExtractorName.OPENMIDNIGHT}-{_HF_REVISION[:8]}-bf16",
        batch_size=8,
        # Do not constrain STAMP's WSI read size: every incoming tile is resized
        # to the model's native 224 px resolution by the transform above.
        input_size_px=None,
        autocast_dtype=torch.bfloat16,
    )
