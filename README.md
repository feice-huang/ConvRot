<p align="center">
  <a href="https://arxiv.org/abs/2512.03673">
    <img src="https://img.shields.io/badge/arXiv-2512.03673-b31b1b.svg">
  </a>
  <a href="https://zhuanlan.zhihu.com/p/1984918169983424471">
    <img src="https://img.shields.io/badge/中文解读-知乎-0084ff.svg">
  </a>
</p>

# ConvRot

> **Note:** I'm not an expert at kernel tuning, so this repository may not be the
> fastest possible implementation — but the accuracy is okay.

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
  per_tensor_scale: true    # dynamic per-tensor FP8 scale for activations
  use_triton_kernel: true   # use the fused rotate+quant Triton kernel
skip_layers:                # substring match on module FQN; these stay bf16
  - x_embedder              #   (e.g. input projections whose in_dim isn't divisible by rot_size,
  - time_text_embed         #    or output-sensitive layers kept for quality)
```

Config layout per model:

- `quantize/convrot_nvfp4.yaml` — convrot config fed to `quantize.py` (all models).
- `infer/convrot_nvfp4.yaml` — convrot config with `quantized_path`, fed to `infer.py` (all models).
- `infer/bf16.yaml` — unquantized baseline, fed to `infer.py` (all models).

## Results

Qualitative comparison on the first three [PRISM-Bench](https://github.com/Anytt/PRISM-Bench)
`text_rendering` prompts across FLUX.1-dev, Qwen-Image and Z-Image-Turbo. Columns
are precision: **bf16** (reference), **fp8**, **nvfp4-convrot** (ConvRot W4A4) and **nvfp4-convrot-selfcalib** (ConvRot W4A4,
self-calibrated static activation scale, faster than nvfp4-convrot).

Prompts:
1. *A compact green circuit board featuring a Raspberry Pi logo, labeled "Raspberry Pi 4 Model B," with Ethernet, USB, and GPIO connectors, along with various electronic components.*
2. *A bottle labeled "WHISTLEPIG" featuring "SMOKED BARREL-AGED RYE" sits alongside two clear whiskey glasses, showcasing a refined presentation of the spirit.*
3. *A silhouette of an airport featuring control towers, cranes, and a terminal building against an orange sunset, with a road sign reading "TOWER!" pointing towards the structures.*

| Model / Prompt | bf16 | fp8 | nvfp4-convrot | nvfp4-convrot-selfcalib |
|:--|:--:|:--:|:--:|:--:|
| **Z-Image-Turbo** #1 | <img src="assets/z-image-turbo_bf16_0.png" width="180"> | <img src="assets/z-image-turbo_fp8_0.png" width="180"> | <img src="assets/z-image-turbo_nvfp4-convrot_0.png" width="180"> | <img src="assets/z-image-turbo_nvfp4-convrot-calibrate_0.png" width="180"> |
| **Z-Image-Turbo** #2 | <img src="assets/z-image-turbo_bf16_1.png" width="180"> | <img src="assets/z-image-turbo_fp8_1.png" width="180"> | <img src="assets/z-image-turbo_nvfp4-convrot_1.png" width="180"> | <img src="assets/z-image-turbo_nvfp4-convrot-calibrate_1.png" width="180"> |
| **Z-Image-Turbo** #3 | <img src="assets/z-image-turbo_bf16_2.png" width="180"> | <img src="assets/z-image-turbo_fp8_2.png" width="180"> | <img src="assets/z-image-turbo_nvfp4-convrot_2.png" width="180"> | <img src="assets/z-image-turbo_nvfp4-convrot-calibrate_2.png" width="180"> |
| **FLUX.1-dev** #1 | <img src="assets/flux_bf16_0.png" width="180"> | <img src="assets/flux_fp8_0.png" width="180"> | <img src="assets/flux_nvfp4-convrot_0.png" width="180"> | <img src="assets/flux_nvfp4-convrot-calibrate_0.png" width="180"> |
| **FLUX.1-dev** #2 | <img src="assets/flux_bf16_1.png" width="180"> | <img src="assets/flux_fp8_1.png" width="180"> | <img src="assets/flux_nvfp4-convrot_1.png" width="180"> | <img src="assets/flux_nvfp4-convrot-calibrate_1.png" width="180"> |
| **FLUX.1-dev** #3 | <img src="assets/flux_bf16_2.png" width="180"> | <img src="assets/flux_fp8_2.png" width="180"> | <img src="assets/flux_nvfp4-convrot_2.png" width="180"> | <img src="assets/flux_nvfp4-convrot-calibrate_2.png" width="180"> |
| **Qwen-Image** #1 | <img src="assets/qwen-image_bf16_0.png" width="180"> | <img src="assets/qwen-image_fp8_0.png" width="180"> | <img src="assets/qwen-image_nvfp4-convrot_0.png" width="180"> | <img src="assets/qwen-image_nvfp4-convrot-calibrate_0.png" width="180"> |
| **Qwen-Image** #2 | <img src="assets/qwen-image_bf16_1.png" width="180"> | <img src="assets/qwen-image_fp8_1.png" width="180"> | <img src="assets/qwen-image_nvfp4-convrot_1.png" width="180"> | <img src="assets/qwen-image_nvfp4-convrot-calibrate_1.png" width="180"> |
| **Qwen-Image** #3 | <img src="assets/qwen-image_bf16_2.png" width="180"> | <img src="assets/qwen-image_fp8_2.png" width="180"> | <img src="assets/qwen-image_nvfp4-convrot_2.png" width="180"> | <img src="assets/qwen-image_nvfp4-convrot-calibrate_2.png" width="180"> |

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



