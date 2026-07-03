"""
Offline quantization entry point: load a bf16 transformer, apply ConvRot+NVFP4,
and save the quantized transformer to disk.

The remaining pipeline components (text encoders, VAE, scheduler, tokenizer)
are *not* duplicated — they're loaded from the original model.path at
inference time.

Usage:
    python quantize.py \
        --config configs/flux_dev/quantize/convrot_nvfp4.yaml \
        --output /path/to/FLUX.1-dev-ConvRot-nvfp4

The output directory is what you put in `model.quantized_path` in an infer
config (e.g. configs/flux_dev/infer/convrot_nvfp4.yaml) to drive inference.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convrot  # registers RotatedNVFP4Tensor / ConvRotNVFP4Config safe globals
from convrot.models import get_model_spec
from infer import build_quant_config, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="YAML for the quantization to apply")
    parser.add_argument("--output", type=str, required=True,
                        help="directory to write the quantized transformer to")
    parser.add_argument("--calibrate", type=int, default=0, metavar="N",
                        help="convrot_nvfp4 only: before saving, run N full "
                             "denoise generations under calibrate() so each "
                             "rotated linear freezes the running-max upper bound "
                             "of its activation scale. The saved model then loads "
                             "single-pass with no first-forward self-calibration. "
                             "0 (default) saves an uncalibrated scale (NaN); the "
                             "first forward at inference self-calibrates.")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the calibration generations")
    args = parser.parse_args()

    torch.backends.cuda.preferred_blas_library("cublaslt")

    cfg = load_config(args.config)
    w_dtype = cfg["weight"]["dtype"]
    if w_dtype == "bfloat16":
        raise ValueError("config has weight.dtype=bfloat16; nothing to quantize")
    model_type = cfg["model"]["type"]
    base_path = cfg["model"]["path"]
    print(f"Quantizing {model_type} ({w_dtype}) from {base_path}")

    spec = get_model_spec(model_type)
    quant_config = build_quant_config(cfg)

    # Load just the transformer with quantization applied. We deliberately bypass
    # the full pipeline because we only need to serialize the quantized weights;
    # text encoders / VAE / scheduler stay on the original disk path.
    from diffusers import PipelineQuantizationConfig
    assert isinstance(quant_config, PipelineQuantizationConfig)
    transformer_quant = quant_config.quant_mapping["transformer"]

    pipeline_cls = spec.pipeline_cls
    t0 = time.perf_counter()
    pipe = pipeline_cls.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        quantization_config=PipelineQuantizationConfig(
            quant_mapping={"transformer": transformer_quant}
        ),
    )
    t1 = time.perf_counter()
    print(f"Loaded and quantized in {t1 - t0:.1f}s")

    # Optional offline calibration. `quant_config.quant_mapping` only touches the
    # transformer, so the pipeline still has real text encoders / VAE / scheduler
    # — we can run a full denoise trajectory. Under calibrate() each rotated
    # linear folds its rotated amax into the shared act_per_tensor_scale via
    # running-max, capturing the upper bound over all timesteps. The frozen scale
    # rides along in the RotatedNVFP4Tensor and is serialized with the weights, so
    # the offline model loads single-pass with no first-forward self-calibration.
    if args.calibrate > 0:
        if w_dtype != "convrot_nvfp4":
            raise ValueError(
                f"--calibrate is only meaningful for convrot_nvfp4 "
                f"(config has weight.dtype={w_dtype}); there is no static "
                f"activation scale to calibrate"
            )
        from convrot.rotated_nvfp4_tensor import calibrate
        base_inputs = spec.build_inputs(cfg.get("inputs", {}), seed=args.seed)
        print(f"Calibrating activation scale over {args.calibrate} generation(s) "
              f"({base_inputs.get('num_inference_steps', '?')} steps each)...")
        tc0 = time.perf_counter()
        with torch.inference_mode(), calibrate():
            for i in range(args.calibrate):
                pipe(**dict(base_inputs, generator=torch.manual_seed(args.seed)))
                print(f"  calib gen {i + 1}/{args.calibrate}", flush=True)
        print(f"Calibrated in {time.perf_counter() - tc0:.1f}s")

    transformer = pipe.transformer
    os.makedirs(args.output, exist_ok=True)
    print(f"Saving quantized transformer to {args.output} ...")
    t2 = time.perf_counter()
    # torchao quantized tensors aren't safetensors-compatible — they pickle through torch.save.
    transformer.save_pretrained(args.output, safe_serialization=False)
    t3 = time.perf_counter()
    print(f"Saved in {t3 - t2:.1f}s")

    # Quick on-disk summary
    total_bytes = 0
    for root, _, files in os.walk(args.output):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))
    print(f"Total on disk: {total_bytes / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
