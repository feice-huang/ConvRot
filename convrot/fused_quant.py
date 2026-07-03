"""
Fused Hadamard-rotate + NVFP4 activation quantization (Triton), single pass.

The ConvRot activation path is  to_nvfp4(hadamard_rotate(x)).  Done naively that
is: a standalone rotate kernel (read x, write rotated bf16), then torchao's
to_nvfp4 (read rotated for amax, read rotated again to quantize, write fp4).
That is ~4 full bf16 passes over the MxK activation.

The pure-NVFP4 target (torchao's to_nvfp4(x)) is just: amax
read + quant read + fp4 write.  The Hadamard rotation flattens the activation
distribution, so a per-tensor scale calibrated once matches the dynamic
per-batch scale to within NVFP4 quant noise.  That lets us skip the amax pass
entirely and quantize in a SINGLE read over the activation:

  rotate(x) on-chip via tl.dot(x, H)  ->  quantize with a static global_scale
                                          ->  fp4 + swizzled block scales (1 read + fp4 write)

matching the pure-NVFP4 target's traffic with the rotate folded in as ~free
tensor-core work.  The static scale is supplied by the caller (calibrated once
and frozen in RotatedNVFP4Tensor.act_per_tensor_scale); this module no longer
computes a runtime amax.

Output is byte-compatible with torchao's swizzled-scale NVFP4Tensor so the
existing _scaled_mm dispatch consumes it unchanged.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

from .ops import _rotation_matrix_bf16

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0


# The two Triton primitives below are vendored verbatim from pytorch/MSLK
# (mslk/quantize/triton/fp4_quantize.py, BSD-3-Clause licensed) so this repo has
# no runtime dependency on mslk. Both are self-contained @triton.jit device
# functions using only triton.language.


@triton.jit
def nvfp4_scale_swizzle(offs_m):
    """
    Produces scale offsets swizzled according to the blackwell 128x4 scale layout.
    Each 128x4 layout can be viewed as 32 4x4 layouts, each of which we'll refer to below as a sub_layout.

    The returned offsets assume a 128x4 layout starting at 0. offs_m could be a subset of rows within a 128x4 layout.
    """
    # Offset of the 4x4 sub_layout within the 128x4 layout
    sub_layout_idx = offs_m % 32
    sub_layout_stride = 16
    sub_layout_off = sub_layout_idx * sub_layout_stride
    # Which row within the 4x4 sub_layout
    sub_layout_row = offs_m // 32
    # Offsets of the elements within 4x4 sub_layout
    elems = tl.arange(0, 4)[None, :]
    sub_layout_elem_offs = sub_layout_row * 4 + elems

    scale_offs = sub_layout_off + sub_layout_elem_offs

    return scale_offs


@triton.jit
def convert_fp32_to_fp4_packed(x_pairs):
    """Convert FP32 pairs to packed FP4 format.

    This function takes tensor where consecutive values along the last dimension
    are packed together into single bytes.

    Args:
        x_pairs: [Tensor, Tensor] both w/ shapes [..., 1] where zipped last dimension contains
                interleaved pairs of FP32 values to be packed together.

    Returns:
        Packed tensor with shape [...] (last dimension removed) where each
        element is an int8 containing 2 FP4 values:
        - First value of pair → low nibble (bits 0-3)
        - Second value of pair → high nibble (bits 4-7)

    Example:
        Input:  [128, 32, 2] containing FP32 pairs
        Output: [128, 32] containing packed FP4 bytes

    """

    x_fp4x2 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 byte0, byte1, byte2, byte3;
        cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
        mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints=("=r,r,r,r,r,r,r,r,r"),
        args=x_pairs,
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )

    return x_fp4x2


def _rotation_matrix(rot_size: int, device_index: int) -> torch.Tensor:
    """[rot, rot] bf16 H with hadamard_rotate(x, rot) == x.view(...,rot) @ H.

    Shared with convrot.ops so the fused path and the standalone matmul rotate
    use one captured transform (validated in tools/rotate_check.py)."""
    return _rotation_matrix_bf16(rot_size, device_index)


@triton.jit
def _fused_rotate_quant_kernel(
    x_ptr, h_ptr, gscale_ptr, q_ptr, s_ptr,
    M, K,
    stride_xm, stride_xn,
    ROT: tl.constexpr, BLOCK_M: tl.constexpr, BK: tl.constexpr,
    NUM_R_BLOCKS: tl.constexpr, SCALE_K: tl.constexpr,
):
    """Rotate one [BLOCK_M, ROT] tile on-chip and NVFP4-quantize it, single pass.

    ROT columns == ROT/16 nvfp4 blocks. Scales stored in the blackwell 128x4
    swizzled layout so torchao's _scaled_mm reads them directly. Mirrors MSLK
    triton_quantize_nvfp4_kernel arithmetic (global_scale = 1/per_tensor_scale).
    The rotate x_tile @ H is contracted over ROT in BK chunks so H never needs
    to be fully resident.
    """
    E4M3_EPS = 1.5258789e-05
    FP4_MAX = 6.0
    FP8_MAX = 448.0
    pid_m = tl.program_id(0)
    pid_r = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, ROT)
    m_mask = offs_m < M

    acc = tl.zeros((BLOCK_M, ROT), dtype=tl.float32)
    for kk in tl.static_range(0, ROT, BK):
        offs_k = kk + tl.arange(0, BK)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (pid_r * ROT + offs_k)[None, :] * stride_xn,
            mask=m_mask[:, None], other=0.0,
        )
        h = tl.load(h_ptr + offs_k[:, None] * ROT + offs_n[None, :])
        acc += tl.dot(x, h)
    rot = acc  # [BLOCK_M, ROT] fp32

    global_scale = tl.load(gscale_ptr)  # = 1 / per_tensor_scale

    NB: tl.constexpr = ROT // 16
    xb = rot.reshape(BLOCK_M, NB, 16)
    block_amax = tl.max(tl.abs(xb), axis=2)  # [BLOCK_M, NB]

    scales = tl.div_rn(block_amax, FP4_MAX) * global_scale
    scales = tl.clamp(scales, E4M3_EPS, FP8_MAX)
    scales_fp8 = scales.to(tl.float8e4nv)  # [BLOCK_M, NB]

    total_scale = tl.div_rn(global_scale, scales_fp8.to(tl.float32)[:, :, None])
    xb = xb * total_scale  # [BLOCK_M, NB, 16]

    # --- store scales in swizzled 128x4 layout ---
    scale_col0 = pid_r * NB
    offs_m_sw = (pid_m * BLOCK_M) % 128 + tl.arange(0, BLOCK_M)[:, None]
    NUM_ELEM_PER_LAYOUT: tl.constexpr = 128 * 4
    for g in tl.static_range(NB // 4):
        col_group = scale_col0 // 4 + g  # which 4-wide scale column block
        layout_off = ((pid_m * BLOCK_M) // 128) * (SCALE_K // 4) * NUM_ELEM_PER_LAYOUT \
            + col_group * NUM_ELEM_PER_LAYOUT
        scale_offs = layout_off + nvfp4_scale_swizzle(offs_m_sw)  # [BLOCK_M, 4]
        sl = tl.arange(0, 4) + g * 4
        s_chunk = tl.gather(scales_fp8, sl[None, :].broadcast_to(BLOCK_M, 4), axis=1)
        tl.store(s_ptr + scale_offs, s_chunk, mask=m_mask[:, None])

    # --- pack to fp4 and store ---
    xq = convert_fp32_to_fp4_packed(xb.reshape(BLOCK_M, ROT // 2, 2).split())  # [BLOCK_M, ROT//2] uint8
    offs_q = pid_r * (ROT // 2) + tl.arange(0, ROT // 2)
    tl.store(
        q_ptr + offs_m[:, None] * (K // 2) + offs_q[None, :],
        xq, mask=m_mask[:, None],
    )


def fused_rotate_nvfp4_quant(x: torch.Tensor, rot_size: int, per_tensor_scale: torch.Tensor):
    """Return (qdata fp4x2 [M,K/2], scales fp8 swizzled flat).

    Single-pass rotate+quant with a caller-supplied static per_tensor_scale.
    Matches torchao to_nvfp4(hadamard_rotate(x), per_tensor_scale=...,
    use_triton_kernel=True, is_swizzled_scales=True) but reads x only once.
    Thin wrapper over the opaque custom op so torch.compile treats the triton
    launches as a black box.
    """
    xq, scales = torch.ops.convrot.fused_rotate_nvfp4_quant(x, int(rot_size), per_tensor_scale)
    return xq.view(torch.float4_e2m1fn_x2), scales


@torch.library.custom_op("convrot::fused_rotate_nvfp4_quant", mutates_args=())
def _fused_rotate_nvfp4_quant_op(
    x: torch.Tensor, rot_size: int, per_tensor_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _fused_rotate_nvfp4_quant_impl(x, rot_size, per_tensor_scale)


@_fused_rotate_nvfp4_quant_op.register_fake
def _fused_rotate_nvfp4_quant_fake(x: torch.Tensor, rot_size: int, per_tensor_scale: torch.Tensor):
    orig_shape = x.shape
    K = orig_shape[-1]
    M = 1
    for d in orig_shape[:-1]:
        M *= d

    def round_up(a, b):
        return (a + b - 1) // b * b

    rounded_M = round_up(M, 128)
    rounded_K = round_up(K // 16, 4)
    xq = x.new_empty((M, K // 2), dtype=torch.uint8)
    scales = x.new_empty((rounded_M * rounded_K,), dtype=torch.float8_e4m3fn)
    return xq, scales


def _fused_rotate_nvfp4_quant_impl(x: torch.Tensor, rot_size: int, per_tensor_scale: torch.Tensor):
    """Return (qdata uint8 [M,K/2], scales fp8 swizzled flat).

    uint8 qdata (not the fp4x2 view) so the custom-op boundary carries a plain
    dtype; the Python wrapper re-views it. global_scale = 1 / per_tensor_scale.
    """
    assert x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)
    assert rot_size % 64 == 0, "fused path needs rot_size % 64 == 0 (NB % 4 == 0)"
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    M, K = x2.shape
    x2 = x2.contiguous()
    assert K % rot_size == 0
    assert rot_size % 16 == 0
    H = _rotation_matrix(rot_size, x.device.index)

    BLOCK_M = 64
    BK = min(rot_size, 64)
    num_r_blocks = K // rot_size
    grid = (triton.cdiv(M, BLOCK_M), num_r_blocks)

    # Occupancy-driven warp count (tools/tune_fused.py sweep). Small-K shapes
    # (e.g. FLUX qkv/mlp_up, K=3072) launch few programs and under-fill the 188
    # SMs, so more warps/block raises per-block throughput; large-K shapes
    # already saturate, where fewer warps let more blocks co-reside. Threshold
    # at 16 rotate-blocks (K=4096 @ rot=256) splits the two regimes.
    num_warps = 8 if num_r_blocks <= 16 else 4

    global_scale = per_tensor_scale.reciprocal()

    def round_up(a, b):
        return (a + b - 1) // b * b
    rounded_M = round_up(M, 128)
    scale_K = K // 16
    rounded_K = round_up(scale_K, 4)
    scales = x.new_empty((rounded_M, rounded_K), dtype=torch.float8_e4m3fn)
    xq = x.new_empty((M, K // 2), dtype=torch.uint8)

    _fused_rotate_quant_kernel[grid](
        x2, H, global_scale, xq, scales, M, K, x2.stride(0), x2.stride(1),
        ROT=rot_size, BLOCK_M=BLOCK_M, BK=BK, NUM_R_BLOCKS=num_r_blocks,
        SCALE_K=rounded_K, num_warps=num_warps, num_stages=1,
    )

    return xq, scales.flatten()


def fused_rotate_to_nvfp4(x: torch.Tensor, rot_size: int, per_tensor_scale: torch.Tensor):
    """Fused single-pass rotate+quant returning an NVFP4Tensor ready for _scaled_mm.

    Equivalent to NVFP4Tensor.to_nvfp4(hadamard_rotate(x, rot_size),
    per_tensor_scale=per_tensor_scale, is_swizzled_scales=True,
    use_triton_kernel=True) but reads x only once (no amax pass), using the
    caller's frozen static per_tensor_scale.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        hp_data_dims_to_swizzled_scale_dims_nvfp4,
    )

    orig_shape = x.shape
    K = orig_shape[-1]
    M = 1
    for d in orig_shape[:-1]:
        M *= d
    xq, scales = fused_rotate_nvfp4_quant(x, rot_size, per_tensor_scale)
    scale_M, scale_K = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
    return NVFP4Tensor(
        xq.view(M, K // 2),
        scales.view(scale_M, scale_K),
        16,
        x.dtype,
        per_tensor_scale,
        None,
        True,   # is_swizzled_scales
        True,   # use_triton_kernel
        None,   # act_quant_kwargs (this IS the quantized activation)
    )
