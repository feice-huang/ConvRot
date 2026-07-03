"""
Kernel-level benchmark for the ConvRot Hadamard rotate op.

Measures torch.ops.convrot.hadamard_rotate over a sweep of (batch, rot_size).
Useful for tuning the kernel itself or sanity-checking it after changes.

Usage:
    python tools/bench_kernel.py
    python tools/bench_kernel.py --shapes 256 1024 4096 --rot-size 256
"""

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convrot.ops import hadamard_rotate  # noqa: F401  registers convrot::hadamard_rotate


def _time_ms(fn, warmup: int, iters: int, rounds: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) / iters)
    return statistics.mean(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kernel benchmark for hadamard_rotate")
    parser.add_argument("--shapes", type=int, nargs="+", default=[512, 1024, 2048, 4096],
                        help="Sequence of M (=N=K) values")
    parser.add_argument("--rot-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"config: shapes={args.shapes}, rot_size={args.rot_size}, dtype={args.dtype}, "
          f"warmup={args.warmup}, iters={args.iters}, rounds={args.rounds}")
    print(f"{'shape':>8} | {'ms':>10} | {'GB/s':>10} | {'GFLOP':>8} | {'TFLOPS':>8}")
    print("-" * 60)

    for s in args.shapes:
        m = k = s
        if k % args.rot_size != 0:
            raise ValueError(f"K={k} not divisible by rot_size={args.rot_size}")
        x = torch.randn(m, k, device="cuda", dtype=dtype)

        def fn():
            return torch.ops.convrot.hadamard_rotate(x, args.rot_size)

        with torch.inference_mode():
            ms = _time_ms(fn, args.warmup, args.iters, args.rounds)

        # bytes: read + write of x
        bytes_total = 2 * x.numel() * x.element_size()
        gbps = bytes_total / (ms / 1e3) / 1e9
        # flops: 2 * m * k * rot_size (each block-of-rot_size column does H * vec ~= 2*rot_size^2 ops, normalized over rot_size cols)
        flops = 2.0 * m * k * args.rot_size
        gflop = flops / 1e9
        tflops = flops / (ms / 1e3) / 1e12

        print(f"{s:8d} | {ms:10.5f} | {gbps:10.2f} | {gflop:8.2f} | {tflops:8.3f}")


if __name__ == "__main__":
    main()
