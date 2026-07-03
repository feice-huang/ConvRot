"""Model registry for inference and eval drivers.

Each entry declares:
- pipeline_cls: the diffusers Pipeline subclass to instantiate
- build_inputs(cfg, seed): turn a YAML inputs dict into pipeline call kwargs
- post_load(transformer): optional model-specific patch applied after load

To add a new model, add an entry here and a config under configs/<model>/.
The convrot quantization itself is model-agnostic — it only cares about
nn.Linear weights, with skip_layers controlling which sub-modules stay bf16.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


def _build_flux_dev_inputs(cfg: dict, seed: int = 0) -> dict:
    return {
        "prompt": [cfg["prompt"]],
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
        "generator": torch.manual_seed(seed),
        "num_inference_steps": cfg.get("num_inference_steps", 28),
        "guidance_scale": cfg.get("guidance_scale", 3.5),
        "max_sequence_length": cfg.get("max_sequence_length", 512),
    }


def _build_qwen_image_inputs(cfg: dict, seed: int = 0) -> dict:
    """Qwen-Image text-to-image (no input image). Uses true-CFG, so a negative
    prompt is passed through."""
    return {
        "prompt": [cfg["prompt"]],
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
        "generator": torch.manual_seed(seed),
        "num_inference_steps": cfg.get("num_inference_steps", 50),
        "true_cfg_scale": cfg.get("true_cfg_scale", 4.0),
        "negative_prompt": cfg.get("negative_prompt", ""),
    }


def _build_z_image_inputs(cfg: dict, seed: int = 0) -> dict:
    """Z-Image-Turbo text-to-image. Distilled 8-NFE model: the diffusers doc
    string recommends num_inference_steps=9 (→ 8 DiT forwards) and
    guidance_scale=0.0 (Turbo models run without CFG, so no negative prompt /
    true-CFG double-call)."""
    return {
        "prompt": [cfg["prompt"]],
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
        "generator": torch.manual_seed(seed),
        "num_inference_steps": cfg.get("num_inference_steps", 9),
        "guidance_scale": cfg.get("guidance_scale", 0.0),
        "max_sequence_length": cfg.get("max_sequence_length", 512),
    }


@dataclass
class ModelSpec:
    pipeline_cls: Any
    build_inputs: Callable[[dict, int], dict]
    post_load: Optional[Callable[[Any], None]] = None


def _registry() -> dict[str, ModelSpec]:
    from diffusers import FluxPipeline, QwenImagePipeline, ZImagePipeline

    return {
        "flux_dev": ModelSpec(
            pipeline_cls=FluxPipeline,
            build_inputs=_build_flux_dev_inputs,
            post_load=None,
        ),
        "qwen_image": ModelSpec(
            pipeline_cls=QwenImagePipeline,
            build_inputs=_build_qwen_image_inputs,
            post_load=None,
        ),
        "z_image": ModelSpec(
            pipeline_cls=ZImagePipeline,
            build_inputs=_build_z_image_inputs,
            post_load=None,
        ),
    }


_CACHED: Optional[dict[str, ModelSpec]] = None


def get_model_spec(name: str) -> ModelSpec:
    global _CACHED
    if _CACHED is None:
        _CACHED = _registry()
    if name not in _CACHED:
        raise ValueError(f"Unknown model type: {name!r}. Known: {sorted(_CACHED)}")
    return _CACHED[name]
