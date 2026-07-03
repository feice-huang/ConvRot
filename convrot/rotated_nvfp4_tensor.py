import os

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.utils import TorchAOBaseTensor

from .ops import hadamard_rotate

aten = torch.ops.aten

# Activation rotate strategy. Under torch.compile+cudagraph the matmul rotate
# (convrot.ops.hadamard_rotate, ~1300 GB/s cuBLAS) composes better than the
# opaque fused kernel and is ~6% faster end-to-end on FLUX, so it is the
# default. The fused single-op rotate+quant (no bf16 roundtrip) wins in eager /
# isolation and for callers that don't compile; opt in with CONVROT_FUSED=1.
_USE_FUSED = os.environ.get("CONVROT_FUSED", "0") == "1"


def _quantize_activation(input_tensor, weight_tensor):
    """Rotate + NVFP4-quantize the activation for a rotated-nvfp4 linear.

    Fused path folds the Hadamard rotate into the quant kernel (reads the
    activation twice, matching the pure-NVFP4 target's memory traffic).
    Fallback composes hadamard_rotate + NVFP4Tensor's own act-quant via the
    weight's act_quant_kwargs dispatch. Returns (a_nvfp4, w_nvfp4)."""
    rot_size = weight_tensor.rot_size
    w = weight_tensor.nvfp4_weight
    if _USE_FUSED and rot_size % 64 == 0 and input_tensor.shape[-1] % rot_size == 0:
        from .fused_quant import fused_rotate_to_nvfp4
        a = fused_rotate_to_nvfp4(input_tensor, rot_size)
        return a, w
    # reference: explicit rotate, then let NVFP4 act-quant run in the gemm impl.
    rotated = hadamard_rotate(input_tensor, rot_size)
    return rotated, w


class RotatedNVFP4Tensor(TorchAOBaseTensor):
    tensor_data_names = ["nvfp4_weight"]
    tensor_attribute_names = ["rot_size"]

    def __new__(cls, nvfp4_weight: NVFP4Tensor, rot_size: int):
        if not isinstance(nvfp4_weight, NVFP4Tensor):
            raise TypeError("nvfp4_weight must be NVFP4Tensor")

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            nvfp4_weight.shape,
            dtype=nvfp4_weight.dtype,
            device=nvfp4_weight.device,
            requires_grad=False,
        )
        self.nvfp4_weight = nvfp4_weight
        self.rot_size = int(rot_size)
        return self


implements = RotatedNVFP4Tensor.implements


@implements([aten._to_copy.default])
def rotated_nvfp4_to_copy(func, types, args, kwargs):
    self = args[0]
    if not isinstance(self, RotatedNVFP4Tensor):
        raise NotImplementedError("RotatedNVFP4Tensor: aten._to_copy expects RotatedNVFP4Tensor")

    to_kwargs = self._get_to_kwargs(*args[1:], **kwargs)
    device = to_kwargs["device"]
    dtype = to_kwargs["dtype"]

    moved_weight = self.nvfp4_weight._apply_fn_to_data(lambda x: x.to(device))
    if moved_weight.orig_dtype != dtype:
        moved_weight = NVFP4Tensor(
            moved_weight.qdata,
            moved_weight.scale,
            moved_weight.block_size,
            dtype,
            moved_weight.per_tensor_scale,
            moved_weight.act_per_tensor_scale,
            moved_weight.is_swizzled_scales,
            moved_weight.use_triton_kernel,
            moved_weight.act_quant_kwargs,
        )

    out = RotatedNVFP4Tensor(nvfp4_weight=moved_weight, rot_size=self.rot_size)
    return return_and_correct_aliasing(func, args, kwargs, out)


@implements([aten.t.default])
def rotated_nvfp4_t(func, types, args, kwargs):
    weight_tensor = args[0]
    if not isinstance(weight_tensor, RotatedNVFP4Tensor):
        raise NotImplementedError("RotatedNVFP4Tensor: aten.t expects RotatedNVFP4Tensor")
    return RotatedNVFP4Tensor(
        nvfp4_weight=weight_tensor.nvfp4_weight.t(),
        rot_size=weight_tensor.rot_size,
    )


@implements([torch.nn.functional.linear, aten.linear.default])
def rotated_nvfp4_linear(func, types, args, kwargs):
    input_tensor = args[0]
    weight_tensor = args[1]
    bias = args[2] if len(args) > 2 else None

    if not isinstance(weight_tensor, RotatedNVFP4Tensor):
        raise NotImplementedError("RotatedNVFP4Tensor: weight must be RotatedNVFP4Tensor")

    a, w = _quantize_activation(input_tensor, weight_tensor)
    if isinstance(a, NVFP4Tensor):
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch
        orig_shape = input_tensor.shape
        res = _addmm_nvfp4_dispatch(a, w.t(), func, bias=bias)
        return res.reshape(*orig_shape[:-1], res.shape[-1])
    return torch.nn.functional.linear(a, w, bias)


@implements([aten.mm.default, aten.matmul.default])
def rotated_nvfp4_mm(func, types, args, kwargs):
    input_tensor, weight_tensor = args[0], args[1]
    if not isinstance(weight_tensor, RotatedNVFP4Tensor):
        raise NotImplementedError(
            "RotatedNVFP4Tensor: rhs weight must be RotatedNVFP4Tensor"
        )
    a, w = _quantize_activation(input_tensor, weight_tensor)
    if isinstance(a, NVFP4Tensor):
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch
        return _addmm_nvfp4_dispatch(a, w, func, bias=None)
    return func(a, w)


@implements([aten.addmm.default])
def rotated_nvfp4_addmm(func, types, args, kwargs):
    bias, input_tensor, weight_tensor = args[0], args[1], args[2]
    if not isinstance(weight_tensor, RotatedNVFP4Tensor):
        raise NotImplementedError(
            "RotatedNVFP4Tensor: rhs weight must be RotatedNVFP4Tensor"
        )
    a, w = _quantize_activation(input_tensor, weight_tensor)
    if isinstance(a, NVFP4Tensor):
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch
        return _addmm_nvfp4_dispatch(a, w, func, bias=bias)
    return torch.addmm(bias, a, w)
