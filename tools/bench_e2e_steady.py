"""
Steady-state e2e latency: load once, warm up N times, then time M consecutive
runs in the SAME process. Removes torch.compile first-step re-trace + cudagraph
capture from the measurement, which single-shot infer.py timing includes.

Two compile modes (two-phase workflow — validate quality eager first, then
measure speed compiled):
    eager                        skip torch.compile entirely
    max-autotune-no-cudagraphs   autotuned GEMMs, no cudagraph capture (default)

Usage:
    python tools/bench_e2e_steady.py --config configs/flux_dev/infer/convrot_nvfp4.yaml \
        --compile-mode max-autotune-no-cudagraphs --warmup 2 --iters 10
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import convrot  # noqa: F401  register safe globals
from convrot.models import get_model_spec
from infer import build_quant_config, load_config, load_pipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--compile-mode", type=str, default="max-autotune-no-cudagraphs",
                   choices=["eager", "max-autotune-no-cudagraphs"],
                   help="'eager' skips torch.compile (quality/functional pass); "
                        "'max-autotune-no-cudagraphs' compiles for the speed pass.")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.backends.cuda.preferred_blas_library("cublaslt")

    cfg = load_config(args.config)
    model_type = cfg["model"]["type"]
    w_dtype = cfg["weight"]["dtype"]
    spec = get_model_spec(model_type)
    quant_config = build_quant_config(cfg) if cfg["model"].get("quantized_path") is None else None
    pipe = load_pipeline(cfg, quant_config=quant_config)
    print(f"Config: {args.config}  (model={model_type}, weight={w_dtype})")

    # no-cudagraphs keeps the autotuned GEMMs but skips cudagraph capture, so the
    # raw compiled module is safe to call directly (no static-buffer aliasing).
    if args.compile_mode == "eager":
        print("Eager: torch.compile skipped")
    else:
        pipe.transformer = torch.compile(pipe.transformer, mode=args.compile_mode)
        print(f"Compiled: mode={args.compile_mode}")

    inputs = spec.build_inputs(cfg.get("inputs", {}), seed=args.seed)
    steps = inputs.get("num_inference_steps", 30)

    print(f"Warmup x{args.warmup} (compile trace + cudagraph capture happen here)...")
    with torch.inference_mode():
        for i in range(args.warmup):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pipe(**inputs)
            torch.cuda.synchronize()
            print(f"  warmup {i+1}: {time.perf_counter()-t0:6.2f}s")

    print(f"Timed x{args.iters} (steady state)...")
    times = []
    with torch.inference_mode():
        for i in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pipe(**inputs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)
            print(f"  run {i+1:2d}: {dt:6.3f}s ({dt/steps:.4f}s/step)")

    mean = statistics.mean(times)
    med = statistics.median(times)
    best = min(times)
    print(f"\n=== steady-state ({args.iters} runs, {steps} steps) ===")
    print(f"mean {mean:6.3f}s  median {med:6.3f}s  best {best:6.3f}s  "
          f"({mean/steps:.4f}s/step mean, {best/steps:.4f}s/step best)")


if __name__ == "__main__":
    main()
