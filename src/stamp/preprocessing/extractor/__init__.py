from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from typing import Generic, TypeVar

import torch
from jaxtyping import Float
from PIL import Image
from torch import nn

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


ExtractorModel = TypeVar("ExtractorModel", bound=nn.Module)


@dataclass(frozen=True)
class Extractor(Generic[ExtractorModel]):
    _: KW_ONLY
    model: ExtractorModel
    transform: Callable[[Image.Image], Float[torch.Tensor, "batch ..."]]
    identifier: str
    """An ID _uniquely_ identifying the model and extractor.
    
    If possible, it should include the digest of the model weights etc.
    so that any change in the model also changes its ID.
    """
    batch_size: int = 64
    """Inference batch size used by the preprocessing pipeline."""
    input_size_px: int | None = None
    """Required square tile size, or ``None`` if the extractor accepts any size."""
    input_mpp: float | None = None
    """Required physical input resolution in microns per pixel, if constrained."""
    autocast_dtype: torch.dtype | None = None
    """Optional mixed-precision dtype used for the extractor forward pass."""
