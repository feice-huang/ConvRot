<p align="center">
  <a href="https://arxiv.org/abs/2512.03673">
    <img src="https://img.shields.io/badge/arXiv-2512.03673-b31b1b.svg">
  </a>
  <a href="https://zhuanlan.zhihu.com/p/1984918169983424471">
    <img src="https://img.shields.io/badge/中文解读-知乎-0084ff.svg">
  </a>
</p>

# ConvRot (calibrate)

> **Note:** I'm not an expert at kernel tuning, so this repository may not be the
> fastest possible implementation — but the accuracy is okay.

> **`calibrate` branch: single-pass activation quantization.** On the default
> branch each rotated linear recomputes a *dynamic* per-tensor activation scale
> every forward: an amax pass over the rotated activation, then the quant pass
> (two reads of the activation). This branch replaces that with a *static* scale
> frozen once at quantization time, so every steady-state forward quantizes in a
> **single pass** (no runtime amax), and torch.compile traces a pure
> tensor graph with no data-dependent branch.
>
> The static scale is the running-max **upper bound** of the rotated per-tensor
> amax over a full denoise trajectory: the rotation flattens the activation
> *within* a step, and the amax is stable across prompts at a fixed timestep, but drifts up across timesteps. You can capture the amax offline via `quantize.py --calibrate N` (see below) and 
> serialized into the saved weights; or use an uncalibrated model self-calibrates on its
> first forward. Set `CONVROT_DYNAMIC_ACT=1` to fall back to the
> per-forward dynamic scale (the default branch's behaviour) for A/B comparison.

This repository contains the official implementation of ConvRot, proposed in
*ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers*. ConvRot is a group-wise rotation-based quantization method designed for Diffusion Transformers, enabling W4A4 inference without retraining while preserving visual quality.

Supported workloads:

- **FLUX.1-dev** (`flux_dev`)
- **Qwen-Image** (`qwen_image`)
- **Z-Image-Turbo** (`z_image`) 

You can easily add any model you want by adding `convrot/models.py` and the corresponding `configs`.

## Layout

```
infer.py                 # inference entry point (bf16 / nvfp4 / convrot_nvfp4)
quantize.py              # offline quantization entry point → quantized transformer on disk
convrot/                 # the quantization package
  config.py              #   ConvRotNVFP4Config (torchao AOBaseConfig)
  rotated_nvfp4_tensor.py#   RotatedNVFP4Tensor tensor subclass
  ops.py                 #   convrot::hadamard_rotate custom op (CUDA + matmul paths)
  fused_quant.py         #   fused rotate+NVFP4-quant Triton kernel (activation hot path)
  models.py              #   model registry (pipeline class + input builder per model)
  csrc/                  #   CUDA butterfly-rotation kernel (JIT-compiled on first use)
configs/                 # per-model YAML configs (quantize/ and infer/ per model)
tools/                   # benchmarks (see below)
```

## Requirements

Runs in a environment with [torchao](https://github.com/pytorch/ao) on an NVFP4-capable
GPU (Blackwell / sm_120). Install the pinned deps:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Offline quantization

Produce a quantized transformer on disk from a *quantize*-stage convrot config:

```bash
python quantize.py \
    --config configs/flux_dev/quantize/convrot_nvfp4.yaml \
    --output /path/to/FLUX.1-dev-ConvRot-nvfp4
```

Only the transformer is quantized and serialized; text encoders / VAE / scheduler
are loaded from the original `model.path` at inference time.

**Activation-scale calibration (`--calibrate N`, convrot_nvfp4 only).** This is
how the static single-pass scale (see the note at the top) is frozen offline.
Without `--calibrate`, the saved model carries an *uncalibrated* scale (NaN) and
self-calibrates on its first forward at inference. Passing `--calibrate N` runs
`N` full denoise generations at quantization time, accumulating each rotated
linear's running-max activation scale and freezing it into the saved weights.
The model then loads ready to run single-pass with no first-forward
self-calibration:

```bash
python quantize.py \
    --config configs/qwen_image/quantize/convrot_nvfp4.yaml \
    --output /path/to/Qwen-Image-ConvRot-nvfp4 \
    --calibrate 1
```

### 2. Inference

Point an *infer*-stage convrot config's `model.quantized_path` at the directory
you just wrote, then run:

```bash
python infer.py --config configs/flux_dev/infer/convrot_nvfp4.yaml
python infer.py --config configs/qwen_image/infer/convrot_nvfp4.yaml
```

Useful flags:
- `--compile` / `--compile-mode {default,reduce-overhead,max-autotune}` — torch.compile the transformer.
- `--cpu-offload` — keep idle submodules on CPU (lower VRAM, slower).
- `--out-dir`, `--seed`.

A bf16 config (`configs/<model>/infer/bf16.yaml`) needs no quantization step —
just run `infer.py` on it directly.

## Config reference

```yaml
model:
  type: flux_dev            # registry key: flux_dev | qwen_image | z_image
  path: <base bf16 model>   # HF repo id or local path
  quantized_path: <dir>     # infer stage only: quantized transformer written by quantize.py
inputs:                     # forwarded to the pipeline (prompt, height, steps, ...)
  prompt: "..."
weight:
  dtype: convrot_nvfp4      # bfloat16 | convrot_nvfp4
  rot_size: 256             # Hadamard block size (power of 4, ≤ 1024); K must be divisible by it
activation:
  per_tensor_scale: true    # per-tensor FP8 scale for the NVFP4 activation quant
  use_triton_kernel: true   # use the fused rotate+quant Triton kernel
skip_layers:                # substring match on module FQN; these stay bf16
  - x_embedder              #   (e.g. input projections whose in_dim isn't divisible by rot_size,
  - time_text_embed         #    or output-sensitive layers kept for quality)
```

> On this branch the activation per-tensor scale is **static** (calibrated once,
> then frozen — see the top note); the YAML flags above configure the underlying
> NVFP4 quant, not the calibration. The static behaviour is always on unless
> `CONVROT_DYNAMIC_ACT=1`.

Config layout per model:

- `quantize/convrot_nvfp4.yaml` — convrot config fed to `quantize.py` (all models).
- `infer/convrot_nvfp4.yaml` — convrot config with `quantized_path`, fed to `infer.py` (all models).
- `infer/bf16.yaml` — unquantized baseline, fed to `infer.py` (all models).

## Reference

If you find ConvRot useful in your research, please cite:

```bibtex
@article{huang2025convrot,
  title={ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers},
  author={Huang, Feice and Han, Zuliang and Zhou, Xing and Chen, Yihuang and Zhu, Lifei and Wang, Haoqian},
  journal={arXiv preprint arXiv:2512.03673},
  year={2025}
}
```

