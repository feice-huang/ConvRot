<p align="center">
  <a href="https://arxiv.org/abs/2512.03673">
    <img src="https://img.shields.io/badge/arXiv-2512.03673-b31b1b.svg">
  </a>
  <a href="https://zhuanlan.zhihu.com/p/1984918169983424471">
    <img src="https://img.shields.io/badge/中文解读-知乎-0084ff.svg">
  </a>
</p>

# ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers

This repository contains the official implementation of ConvRot, proposed in
*ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers*.

ConvRot is a group-wise rotation-based quantization method designed for Diffusion Transformers, enabling W4A4 inference without retraining while preserving visual quality.

---

## 🔍 Overview

We propose **ConvRot**, a rotation-based quantization approach that:

* Leverages Regular Hadamard Transform (RHT) to suppress both
  row-wise and column-wise outliers
* Reduces rotation complexity from quadratic to linear
* Is plug-and-play, requiring no retraining or calibration
* Preserves high-fidelity visual generation under 4-bit quantization

Building on ConvRot, we further design **ConvLinear4bit**, a unified module that integrates:

* Rotation
* Quantization
* GEMM
* Dequantization

into a single layer, enabling efficient W4A4 inference for Diffusion Transformers.

---
This codebase is built on top of [QuaRot](https://github.com/spcl/QuaRot).  
ConvRot-related code is located in `QuaRot/convrot` and `QuaRot/e2e` (coming soon).

---

## 🚀 Quick Start

> The following steps mirror the usage in QuaRot.

### Installation

```bash
cd QuaRot
pip install -e .   # or: pip install .
```

### Quantization

```bash
python e2e/quant/regular-256-mix.py
```

### Inference

```bash
python e2e/inference/regular-256-mix.py
```

---

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{huang2025convrot,
  title={ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers},
  author={Huang, Feice and Han, Zuliang and Zhou, Xing and Chen, Yihuang and Zhu, Lifei and Wang, Haoqian},
  journal={arXiv preprint arXiv:2512.03673},
  year={2025}
}
```

