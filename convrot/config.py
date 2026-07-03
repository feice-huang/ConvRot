from dataclasses import dataclass

import torch
import torch.nn as nn
from torchao.core.config import AOBaseConfig
from torchao.quantization import register_quantize_module_handler
from torchao.quantization.quant_api import _QUANTIZE_CONFIG_HANDLER
from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig

from .ops import hadamard_rotate
from .rotated_nvfp4_tensor import RotatedNVFP4Tensor


@dataclass
class ConvRotNVFP4Config(AOBaseConfig):
    rot_size: int = 256
    use_dynamic_per_tensor_scale: bool = True
    use_triton_kernel: bool = True


@register_quantize_module_handler(ConvRotNVFP4Config)
def _convrot_nvfp4_transform(module: nn.Module, config: ConvRotNVFP4Config):
    if not isinstance(module, nn.Linear):
        return module

    if module.weight.device.type != "cuda":
        return module

    rot_size = int(config.rot_size)
    weight = module.weight.data
    if weight.shape[1] % rot_size != 0:
        raise ValueError(
            f"weight.shape[1]={weight.shape[1]} is not divisible by rot_size={rot_size}; "
            f"add this layer to skip_layers"
        )

    rotated_weight = hadamard_rotate(weight, rot_size)
    module.weight.data = rotated_weight

    base_cfg = NVFP4DynamicActivationNVFP4WeightConfig(
        use_triton_kernel=config.use_triton_kernel,
        use_dynamic_per_tensor_scale=config.use_dynamic_per_tensor_scale,
    )
    handler = _QUANTIZE_CONFIG_HANDLER[type(base_cfg)]
    module = handler(module, base_cfg)

    if isinstance(module.weight, nn.Parameter):
        nvfp4_weight = module.weight.detach()
    else:
        nvfp4_weight = module.weight

    if not torch.is_tensor(nvfp4_weight):
        raise TypeError("Expected tensor-like weight after NVFP4 quantization")

    # Persistent 1-element calibration buffer for the static activation scale
    # (init NaN = "uncalibrated"). Under convrot.rotated_nvfp4_tensor.calibrate()
    # each forward folds the rotated amax into it via running-max, capturing the
    # upper bound over a full denoise trajectory; an offline --calibrate pass or
    # the eager warmup fills it before compile/serialization. If still NaN at
    # steady state, the first forward self-calibrates as a fallback. Once set,
    # every forward quantizes the rotated activation single-pass (no amax pass).
    act_per_tensor_scale = torch.full(
        (1,), float("nan"), device=nvfp4_weight.device, dtype=torch.float32
    )

    module.weight = nn.Parameter(
        RotatedNVFP4Tensor(
            nvfp4_weight=nvfp4_weight,
            rot_size=rot_size,
            act_per_tensor_scale=act_per_tensor_scale,
        ),
        requires_grad=False,
    )
    return module
