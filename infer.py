import argparse
import os
import time

import yaml
import torch

# Persist inductor's compile cache across runs so re-running the same model+shape
# combination skips codegen and reuses the captured cudagraph keys.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "torchinductor"))
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")

from diffusers import PipelineQuantizationConfig, TorchAoConfig
import convrot  # registers RotatedNVFP4Tensor / ConvRotNVFP4Config safe globals
from convrot import ConvRotNVFP4Config
from convrot.models import get_model_spec


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_quant_config(cfg):
    w_dtype = cfg["weight"]["dtype"]
    skip = cfg.get("skip_layers", [])

    if w_dtype == "bfloat16":
        return None

    if w_dtype == "convrot_nvfp4":
        act_cfg = cfg.get("activation", {})
        ao_config = ConvRotNVFP4Config(
            rot_size=cfg["weight"].get("rot_size", 256),
            use_dynamic_per_tensor_scale=act_cfg.get("per_tensor_scale", True),
            use_triton_kernel=act_cfg.get("use_triton_kernel", True),
        )
    else:
        raise ValueError(f"Unknown weight dtype: {w_dtype}")

    return PipelineQuantizationConfig(
        quant_mapping={"transformer": TorchAoConfig(
            ao_config,
            modules_to_not_convert=skip,
        )}
    )


def load_pipeline(cfg, quant_config=None, cpu_offload=False):
    """Build a pipeline for the given config.

    Three paths:
    - quantized_path set → offline: load the quantized transformer from disk,
      load the rest of the pipeline from base path. No online quantization.
    - quant_config given → online: load bf16, quantize during load.
    - neither → plain bf16.

    When cpu_offload is True the pipeline is not moved to "cuda" and device_map
    is not used; instead diffusers' enable_model_cpu_offload manages placement,
    keeping idle submodules on CPU and streaming each onto the GPU on demand.
    """
    spec = get_model_spec(cfg["model"]["type"])
    path = cfg["model"]["path"]
    quantized_path = cfg["model"].get("quantized_path")

    if quantized_path is not None:
        if quant_config is not None:
            raise ValueError(
                "model.quantized_path and quantization_config are mutually exclusive — "
                "the quantized transformer is already on disk"
            )
        # Import the transformer class via the pipeline's documented config so
        # we don't hardcode it per model. diffusers' Pipeline._load_pretrained_model
        # already resolves this from model_index.json; we mirror that by reading
        # the saved transformer's config.json and dispatching through diffusers.
        from diffusers import models as _diffusers_models
        import json
        with open(os.path.join(quantized_path, "config.json"), "r") as f:
            tcfg = json.load(f)
        transformer_cls = getattr(_diffusers_models, tcfg["_class_name"])
        transformer = transformer_cls.from_pretrained(
            quantized_path, torch_dtype=torch.bfloat16, use_safetensors=False,
        )
        if not cpu_offload:
            transformer = transformer.to("cuda")
        pipe = spec.pipeline_cls.from_pretrained(
            path, transformer=transformer, torch_dtype=torch.bfloat16,
        )
        if not cpu_offload:
            pipe = pipe.to("cuda")
    elif quant_config is not None:
        # device_map="cuda" loads everything resident on the GPU and is mutually
        # exclusive with cpu offload; drop it when offloading so the offload hooks
        # own device placement.
        kwargs = dict(torch_dtype=torch.bfloat16, quantization_config=quant_config)
        if not cpu_offload:
            kwargs["device_map"] = "cuda"
        pipe = spec.pipeline_cls.from_pretrained(path, **kwargs)
    else:
        pipe = spec.pipeline_cls.from_pretrained(path)
        pipe.to(torch.bfloat16)
        if not cpu_offload:
            pipe.to("cuda")

    if cpu_offload:
        pipe.enable_model_cpu_offload()

    pipe.set_progress_bar_config(disable=None)
    if spec.post_load is not None:
        spec.post_load(pipe.transformer)
    return pipe


def _wrap_cudagraph_safe(compiled_transformer):
    """Wrap a compiled transformer so each call marks a cudagraph step boundary and
    returns a cloned output. Required when the pipeline holds multiple compiled outputs
    live simultaneously (e.g. CFG cond+uncond)."""
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    inner = compiled_transformer

    def forward(*args, **kwargs):
        torch.compiler.cudagraph_mark_step_begin()
        out = inner(*args, **kwargs)
        if isinstance(out, Transformer2DModelOutput):
            return Transformer2DModelOutput(sample=out.sample.clone())
        if isinstance(out, tuple):
            return tuple(x.clone() if torch.is_tensor(x) else x for x in out)
        if torch.is_tensor(out):
            return out.clone()
        return out

    # Diffusers' pipeline calls `pipeline.transformer(...)` directly; expose the same
    # interface and forward attribute lookups (e.g. .config) to the wrapped module.
    class _Wrapped:
        def __init__(self, mod):
            object.__setattr__(self, "_mod", mod)
        def __call__(self, *args, **kwargs):
            return forward(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_mod"), name)

    return _Wrapped(inner)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion model quantized inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a per-model YAML (e.g. configs/flux_dev/infer/convrot_nvfp4.yaml)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the transformer before inference")
    parser.add_argument("--compile-mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (only used when --compile is set)")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="use diffusers' enable_model_cpu_offload to keep idle "
                             "submodules on CPU (lower VRAM, slower)")
    parser.add_argument("--seed", type=int, default=0,
                        help="diffusion generator seed")
    parser.add_argument("--out-dir", type=str, default="runs",
                        help="directory to save the output image")
    args = parser.parse_args()

    cfg = load_config(args.config)
    w_dtype = cfg["weight"]["dtype"]
    model_type = cfg["model"]["type"]
    print(f"Config: {args.config}  (model={model_type}, weight={w_dtype})")

    spec = get_model_spec(model_type)
    quant_config = build_quant_config(cfg) if cfg["model"].get("quantized_path") is None else None
    pipeline = load_pipeline(cfg, quant_config=quant_config, cpu_offload=args.cpu_offload)
    print("Pipeline loaded.\n")

    if args.compile:
        if args.cpu_offload and args.compile_mode in ("reduce-overhead", "max-autotune"):
            raise ValueError(
                "--cpu-offload is incompatible with cudagraph compile modes "
                "(reduce-overhead/max-autotune): offload hooks move modules across "
                "devices between steps, breaking the static cudagraph buffers. "
                "Use --compile-mode default with --cpu-offload."
            )
        print(f"Compiling transformer with mode='{args.compile_mode}'...")
        compiled = torch.compile(pipeline.transformer, mode=args.compile_mode)
        if args.compile_mode in ("reduce-overhead", "max-autotune"):
            # cudagraphs reuse the output buffer between invocations; the pipeline calls
            # the transformer twice per step (CFG cond+uncond) and references both outputs
            # afterwards. Mark the step boundary and clone outputs so neither alias is overwritten.
            pipeline.transformer = _wrap_cudagraph_safe(compiled)
        else:
            pipeline.transformer = compiled

    inputs = spec.build_inputs(cfg.get("inputs", {}), seed=args.seed)
    steps = inputs.get("num_inference_steps", 30)

    print("Running pipeline...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = pipeline(**inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    out_name = f"{model_type}_{w_dtype.upper()}_output.png"
    out_path = os.path.join(args.out_dir, out_name)
    os.makedirs(args.out_dir, exist_ok=True)
    output.images[0].save(out_path)
    print(f"Image saved to {out_path}")
    print(f"Inference time: {elapsed:.2f}s ({elapsed/steps:.3f}s/step over {steps} steps)")
