from .config import ConvRotNVFP4Config
from .rotated_nvfp4_tensor import RotatedNVFP4Tensor
from . import fused_quant  # noqa: F401  register convrot::fused_rotate_nvfp4_quant op

import torch
from torchao.core.config import ALLOWED_AO_MODULES

ALLOWED_AO_MODULES.update({"convrot", "convrot.config"})

torch.serialization.add_safe_globals([ConvRotNVFP4Config, RotatedNVFP4Tensor])

__all__ = ["ConvRotNVFP4Config", "RotatedNVFP4Tensor"]
