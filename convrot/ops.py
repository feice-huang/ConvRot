"""
Custom ops for ConvRot+NVFP4.

`hadamard_rotate(x, rot_size)` applies the fixed block-Hadamard rotation used by
ConvRot. Two implementations coexist:

- The CUDA butterfly kernel (convrot_ext) is the canonical definition of the
  transform and the ground truth the correctness harness snapshots.
- A tensor-core matmul  x.view(...,rot) @ H  where H is that transform captured
  as a [rot, rot] matrix. On Blackwell this is ~1.4x the butterfly kernel's
  bandwidth (memory-bound; the matmul reaches ~1300 GB/s vs ~850 GB/s) and is
  what the public op uses whenever H is available (power-of-4 rot <= 1024).

H is built once per (rot, device) from the CUDA kernel itself, so the matmul
path reproduces exactly the committed transform (validated in
tools/rotate_check.py). The full rotated linear is composed in Python:
    F.linear(hadamard_rotate(x, rot_size), nvfp4_weight, bias)
letting NVFP4Tensor's dispatch handle the gemm; the activation hot path is
instead fused (see convrot.fused_quant).
"""

import functools
import os

import torch
from torch.utils.cpp_extension import load

_EXT = None


def _get_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    build_dir = "/tmp/torch_extensions/convrot_ext"
    os.makedirs(build_dir, exist_ok=True)
    _EXT = load(
        name="convrot_ext",
        sources=[
            "convrot/csrc/bindings.cpp",
            "convrot/csrc/kernels.cu",
        ],
        build_directory=build_dir,
        verbose=False,
    )
    return _EXT


def _is_power_of_four(v: int) -> bool:
    if v < 4:
        return v == 1  # rot==1 is a no-op identity; <4 & !=1 unsupported
    while v % 4 == 0:
        v //= 4
    return v == 1


@functools.lru_cache(maxsize=16)
def _rotation_matrix_bf16(rot_size: int, device_index: int) -> torch.Tensor:
    """[rot, rot] bf16 H with  cuda_kernel(x) == (x.view(...,rot) @ H).view(...).

    Captured from the CUDA butterfly kernel by rotating a scaled identity (the
    C=256 factor keeps the +/-1 combinations exact in bf16), so the matmul path
    is byte-for-byte the committed transform up to bf16 matmul rounding."""
    device = torch.device("cuda", device_index)
    C = 256.0
    eye = torch.eye(rot_size, device=device, dtype=torch.bfloat16) * C
    H = _get_ext().hadamard_rotate_cuda(eye, int(rot_size)).to(torch.float32) / C
    return H.to(torch.bfloat16).contiguous()


def _can_use_matmul(rot_size: int) -> bool:
    return 4 <= rot_size <= 1024 and _is_power_of_four(rot_size)


@torch.library.custom_op("convrot::hadamard_rotate", mutates_args=())
def hadamard_rotate(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    rot_size = int(rot_size)
    if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and _can_use_matmul(rot_size):
        K = x.shape[-1]
        if K % rot_size == 0:
            H = _rotation_matrix_bf16(rot_size, x.device.index).to(x.dtype)
            xb = x.reshape(*x.shape[:-1], K // rot_size, rot_size)
            return torch.matmul(xb, H).reshape(x.shape).contiguous()
    return _get_ext().hadamard_rotate_cuda(x, rot_size)


@hadamard_rotate.register_fake
def _hadamard_rotate_fake(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    return torch.empty_like(x)
