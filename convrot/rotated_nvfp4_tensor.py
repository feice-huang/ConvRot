import os

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.utils import TorchAOBaseTensor

from .ops import hadamard_rotate

aten = torch.ops.aten

# Activation rotate strategy. The fused single-op rotate+quant (single pass, no
# bf16 roundtrip) is the default. Under torch.compile (max-autotune-no-cudagraphs)
# on FLUX it holds a rock-steady 0.1564 s/step, within ~20% of the pure-NVFP4
# ceiling (0.130); the matmul-rotate path is 0.1554 s/step (clean back-to-back
# A/B, both at full 2422 MHz — the two are within 0.6%, i.e. equivalent under
# compile). Fused wins as default because in EAGER it is clearly faster
# (layer bench 1.55x vs matmul-rotate 1.24x vs bf16) and is a single kernel with
# no standalone-rotate intermediate. Opt into matmul-rotate with CONVROT_FUSED=0
# for debugging. (Fused requires rot%64==0 and K%rot==0, else auto-falls back.)
_USE_FUSED = os.environ.get("CONVROT_FUSED", "1") == "1"

# Diagnostic override: recompute the per-tensor activation scale dynamically on
# every forward (the pre-refactor behaviour) instead of freezing a static scale
# calibrated on the first forward. Used to isolate whether static-scale
# calibration is the source of end-to-end quality loss on diffusion, where the
# activation distribution drifts across timesteps.
_DYNAMIC_ACT = os.environ.get("CONVROT_DYNAMIC_ACT", "0") == "1"

# Static activation-scale calibration.
#
# The rotated per-tensor amax drifts up across diffusion timesteps (probe: up to
# ~3.3x vs step0) but is stable across inputs at a fixed timestep (cross-prompt
# CV ~6.5%). So the correct static scale is the running-max *upper bound* over a
# full denoise trajectory; freezing it removes the runtime amax pass and lets
# every steady-state forward quantize single-pass with no timestep clipping.
#
# Calibration is a distinct phase from steady-state inference, controlled by a
# module-level Python flag (not a tensor attribute or a per-forward host read):
#   * calibrating: each forward measures the rotated amax and folds it into the
#     shared act_per_tensor_scale via running-max (two-pass, eager only).
#   * steady state: forwards read the frozen scale and run single-pass, with no
#     .item()/isnan/data-dependent branch — so torch.compile traces a pure
#     tensor graph (the calibration block is constant-folded away).
#
# The two-phase workflow drives this: run a few eager warmup forwards under
# calibrate() (or an offline --calibrate pass), then freeze and compile. If a
# scale is still uncalibrated (NaN) at steady state we self-calibrate that one
# forward as a fallback so bare eager use "just works".
_CALIBRATING = False


def calibrate():
    """Context manager: forwards accumulate the running-max activation scale."""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        global _CALIBRATING
        prev = _CALIBRATING
        _CALIBRATING = True
        try:
            yield
        finally:
            _CALIBRATING = prev

    return _cm()


def is_calibrating():
    return _CALIBRATING


def _is_traced_tensor(*tensors):
    """True if we're inside a torch.compile trace (fake/functional/proxy modes).

    Calibration does host-side reads (isnan/.any()/bool) that are only valid on
    real eager tensors; under compile the subclass is decomposed with fake or
    functional tensors where those reads become data-dependent guards. We detect
    the trace by an active fake/proxy dispatch mode or a functional tensor arg,
    and skip calibration entirely so the compiled graph is a pure single-pass
    quant with the already-frozen scale.
    """
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import FunctionalTensor

    for t in tensors:
        if isinstance(t, (FakeTensor, FunctionalTensor)):
            return True
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is not None:
        return True
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is not None:
        return True
    return False


def _quantize_activation(input_tensor, weight_tensor):
    """Rotate + NVFP4-quantize the activation for a rotated-nvfp4 linear.

    Static-scale single-pass path. The Hadamard rotation flattens the activation
    *within* a forward, and the probe (tools/probe_amax.py) shows the rotated
    per-tensor amax is stable across inputs at a fixed diffusion timestep
    (cross-prompt CV ~6.5%) but drifts up to ~3.3x across timesteps. A single
    scale frozen on the *first* forward therefore under-scales later timesteps
    and clips large values. We instead freeze the *upper bound* over a
    calibration window: under the ``calibrate()`` context manager
    ``act_per_tensor_scale`` accumulates the running max of the rotated amax
    across every forward (a full denoise trajectory), then freezes when the
    context exits. Every subsequent forward reads the frozen bound and runs
    single-pass, with no timestep clipping.

    Modes:
      * default: static upper-bound scale (single-pass after calibration).
      * CONVROT_DYNAMIC_ACT=1: recompute the scale dynamically every forward
        (pre-refactor behaviour, two-pass) — diagnostic / accuracy ceiling.

    Returns (a_nvfp4, w_nvfp4).
    """
    rot_size = weight_tensor.rot_size
    w = weight_tensor.nvfp4_weight
    aps = weight_tensor.act_per_tensor_scale

    fused_ok = (
        _USE_FUSED
        and rot_size % 64 == 0
        and input_tensor.shape[-1] % rot_size == 0
    )

    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    # Dynamic diagnostic: measure amax and quantize with a fresh scale each call.
    if _DYNAMIC_ACT or aps is None:
        rotated = hadamard_rotate(input_tensor, rot_size)
        orig_shape = rotated.shape
        rotated2d = rotated.reshape(-1, orig_shape[-1])
        pts = per_tensor_amax_to_scale(torch.max(torch.abs(rotated2d)))
        a = NVFP4Tensor.to_nvfp4(
            rotated2d,
            block_size=16,
            per_tensor_scale=pts,
            is_swizzled_scales=True,
            use_triton_kernel=w.use_triton_kernel,
        )
        return a, w

    # Calibration phase (eager only): fold this forward's rotated amax into the
    # shared act_per_tensor_scale via running-max. Also self-calibrate a still-
    # uncalibrated (NaN) scale as a fallback for bare eager use. All host-side
    # reads are gated behind a real-tensor check: under torch.compile the weight
    # subclass is decomposed with Fake/Functional tensors during aot tracing
    # (where torch.compiler.is_compiling() no longer reads True), and an
    # isnan/.any()/bool() on those becomes a data-dependent guard. Skipping the
    # whole block for traced tensors keeps the compiled steady state a pure
    # single-pass quant with the frozen scale.
    if not _is_traced_tensor(aps, input_tensor) and (
        _CALIBRATING or bool(torch.isnan(aps).any())
    ):
        rotated = hadamard_rotate(input_tensor, rot_size)
        amax = torch.max(torch.abs(rotated))
        new_scale = per_tensor_amax_to_scale(amax).reshape(aps.shape)
        if bool(torch.isnan(aps).any()):
            aps.copy_(new_scale)
        else:
            aps.copy_(torch.maximum(aps, new_scale))

    if fused_ok:
        from .fused_quant import fused_rotate_to_nvfp4

        a = fused_rotate_to_nvfp4(input_tensor, rot_size, per_tensor_scale=aps)
        return a, w
    # matmul-rotate reference path, single-pass with the frozen static scale.
    rotated = hadamard_rotate(input_tensor, rot_size)
    orig_shape = rotated.shape
    rotated2d = rotated.reshape(-1, orig_shape[-1])
    a = NVFP4Tensor.to_nvfp4(
        rotated2d,
        block_size=16,
        per_tensor_scale=aps,
        is_swizzled_scales=True,
        use_triton_kernel=w.use_triton_kernel,
    )
    return a, w


class RotatedNVFP4Tensor(TorchAOBaseTensor):
    tensor_data_names = ["nvfp4_weight"]
    tensor_attribute_names = ["rot_size"]
    optional_tensor_data_names = ["act_per_tensor_scale"]

    def __new__(cls, nvfp4_weight: NVFP4Tensor, rot_size: int,
                act_per_tensor_scale=None):
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
        self.act_per_tensor_scale = act_per_tensor_scale
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

    aps = self.act_per_tensor_scale
    aps = aps.to(device) if aps is not None else None
    out = RotatedNVFP4Tensor(
        nvfp4_weight=moved_weight, rot_size=self.rot_size,
        act_per_tensor_scale=aps,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@implements([aten.t.default])
def rotated_nvfp4_t(func, types, args, kwargs):
    weight_tensor = args[0]
    if not isinstance(weight_tensor, RotatedNVFP4Tensor):
        raise NotImplementedError("RotatedNVFP4Tensor: aten.t expects RotatedNVFP4Tensor")
    return RotatedNVFP4Tensor(
        nvfp4_weight=weight_tensor.nvfp4_weight.t(),
        rot_size=weight_tensor.rot_size,
        act_per_tensor_scale=weight_tensor.act_per_tensor_scale,
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
