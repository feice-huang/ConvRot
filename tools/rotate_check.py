"""
Correctness + perf harness for the ConvRot hadamard_rotate kernel.

The transform is defined by whatever the *current committed* kernel computes.
We snapshot it as a ground-truth matrix H (rot_size x rot_size) by applying the
kernel to the identity, save it, and any kernel rewrite must reproduce
y == x @ H^T (within fp tolerance) for random inputs at all target shapes.

Usage:
    python tools/rotate_check.py --make-ref     # snapshot H from current .so
    python tools/rotate_check.py                # validate + time current kernel
"""

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convrot.ops import _get_ext, hadamard_rotate


def _raw_rotate(x, rot):
    """Ground-truth CUDA butterfly kernel, bypassing the matmul fast path."""
    return _get_ext().hadamard_rotate_cuda(x, int(rot))

REF_DIR = "/tmp/convrot_ref"

# (K, M) shapes to validate/time; K must be divisible by rot_size.
SHAPES = [
    (3072, 4608),
    (12288, 4608),
    (15360, 4608),
    (3072, 4096),
    (12288, 1024),
]
ROTS = [16, 64, 256]


def ref_matrix(rot_size, dtype, device):
    """H[i,j] = kernel(e_j)[i]; capture the exact linear map the kernel applies.
    Built in bf16 (the kernel's supported dtype) then promoted; block Hadamard
    values are small integers/inv_norm so bf16 identity columns are exact enough
    to recover the map, but we scale each column by a large constant to expose
    the full-precision structure and divide it back out."""
    C = 256.0  # scale so bf16 mantissa captures the +/-1 combination exactly
    eye = (torch.eye(rot_size, device=device, dtype=torch.bfloat16) * C)
    H = _raw_rotate(eye, rot_size).to(torch.float32) / C
    return H


def make_ref():
    os.makedirs(REF_DIR, exist_ok=True)
    for rot in ROTS:
        H = ref_matrix(rot, torch.float32, "cuda").cpu()
        torch.save(H, os.path.join(REF_DIR, f"H_{rot}.pt"))
        print(f"saved ref H_{rot} shape={tuple(H.shape)}")


def load_ref(rot, device, dtype):
    p = os.path.join(REF_DIR, f"H_{rot}.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing; run --make-ref against a known-good build")
    return torch.load(p).to(device=device, dtype=dtype)


def apply_ref(x, H, rot):
    # x: [M, K], apply H per rot-block along K.
    M, K = x.shape
    xb = x.reshape(M, K // rot, rot).to(torch.float32)
    # H[i,:] = transform(e_i); linear => transform(x) = sum_i x[i]*H[i,:] = x @ H
    yb = torch.einsum("mbr,rs->mbs", xb, H.to(torch.float32))
    return yb.reshape(M, K)


def validate():
    torch.manual_seed(0)
    max_err = 0.0
    for rot in ROTS:
        H = load_ref(rot, "cuda", torch.float32)
        for K, M in SHAPES:
            if K % rot:
                continue
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            got = hadamard_rotate(x, rot).to(torch.float32)
            exp = apply_ref(x, H, rot)
            denom = exp.abs().max().clamp_min(1e-6)
            rel = (got - exp).abs().max() / denom
            max_err = max(max_err, rel.item())
            tag = "OK " if rel < 2e-2 else "BAD"
            print(f"[{tag}] rot={rot:4d} M={M:5d} K={K:6d}  rel_err={rel.item():.2e}")
    print(f"\nmax rel err = {max_err:.2e}  ({'PASS' if max_err < 2e-2 else 'FAIL'})")
    return max_err < 2e-2


def _time_us(fn, warmup=30, iters=200, rounds=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(rounds):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        vals.append(s.elapsed_time(e) / iters * 1000)
    return min(vals)


def perf():
    torch.manual_seed(0)
    rot = 256
    print(f"\nperf (rot={rot}), us, and effective BW (2*M*K*2 bytes):")
    for K, M in SHAPES:
        if K % rot:
            continue
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            t = _time_us(lambda: hadamard_rotate(x, rot))
        gb = 2 * M * K * 2 / 1e9
        print(f"  M={M:5d} K={K:6d}  {t:8.2f} us  {gb/(t/1e6):7.0f} GB/s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--make-ref", action="store_true")
    p.add_argument("--perf-only", action="store_true")
    args = p.parse_args()
    if args.make_ref:
        make_ref()
        return
    ok = True
    if not args.perf_only:
        ok = validate()
    perf()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
