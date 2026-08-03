"""Checkpoint and shape utilities for MiniMax H3."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from musubi_tuner.minimax_h3.model import MiniMaxH3Model
from musubi_tuner.modules.int8_optimization_utils import (
    Int8ConvRotConfig,
    apply_int8_convrot_monkey_patch,
)
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

VIDEO_FPS = 24
AUDIO_SAMPLE_RATE = 32_000
AUDIO_LATENTS_PER_SECOND = 40
VIDEO_FRAME_STRIDE = 17
VIDEO_FRAME_OFFSET = 5


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformerCheckpointLayout:
    int8_convrot_layers: dict[str, Int8ConvRotConfig]
    adaln_curve_grid: int | None = None
    adaln_curve_dim: int | None = None


def is_valid_frame_count(frame_count: int) -> bool:
    return frame_count >= VIDEO_FRAME_OFFSET and (frame_count - VIDEO_FRAME_OFFSET) % VIDEO_FRAME_STRIDE == 0


def align_frame_count(frame_count: int, mode: str = "down") -> int:
    """Align a source frame count to H3's ``17*n+5`` video contract."""

    if mode not in {"down", "up"}:
        raise ValueError("mode must be 'down' or 'up'")
    if frame_count <= VIDEO_FRAME_OFFSET:
        return VIDEO_FRAME_OFFSET
    steps = (frame_count - VIDEO_FRAME_OFFSET) / VIDEO_FRAME_STRIDE
    steps = math.floor(steps) if mode == "down" else math.ceil(steps)
    return VIDEO_FRAME_OFFSET + steps * VIDEO_FRAME_STRIDE


def video_latent_length(frame_count: int) -> int:
    if not is_valid_frame_count(frame_count):
        raise ValueError(f"H3 frame count must be 17*n+5, got {frame_count}")
    return ((frame_count - VIDEO_FRAME_OFFSET) // VIDEO_FRAME_STRIDE) * 5 + 2


def audio_latent_length(frame_count: int) -> int:
    return round(frame_count * AUDIO_LATENTS_PER_SECOND / VIDEO_FPS)


def _index_files(index_path: Path) -> list[Path]:
    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    return [index_path.parent / name for name in dict.fromkeys(weight_map.values())]


def resolve_safetensor_files(path: str, component: str | None = None) -> list[Path]:
    """Resolve a Comfy single file or an official sharded component directory."""

    candidate = Path(path).expanduser()
    if candidate.is_file():
        if candidate.suffix != ".safetensors":
            raise ValueError(f"Expected a .safetensors checkpoint, got {candidate}")
        return [candidate]
    if not candidate.is_dir():
        raise FileNotFoundError(path)

    roots = [candidate]
    if component:
        roots = [
            candidate / "FL2VA" / component,
            candidate / component,
            candidate,
        ]
        if component == "video_vae":
            roots = [root / "source" for root in roots] + roots

    for root in roots:
        if not root.is_dir():
            continue
        indexes = sorted(root.glob("*.safetensors.index.json"))
        if indexes:
            return _index_files(indexes[0])
        files = sorted(root.glob("*.safetensors"))
        if files:
            return files
    raise FileNotFoundError(f"No safetensors weights found for {component or 'model'} under {candidate}")


def inspect_transformer_checkpoint(files: Iterable[Path]) -> TransformerCheckpointLayout:
    """Read H3 shape and Comfy quantization metadata without loading large weights."""

    int8_layers: dict[str, Int8ConvRotConfig] = {}
    curve_grid = None
    curve_dim = None
    for filename in files:
        with MemoryEfficientSafeOpen(str(filename)) as reader:
            if "adaln_t_table" in reader.header:
                shape = tuple(reader.header["adaln_t_table"]["shape"])
                if len(shape) != 2 or shape[0] < 2:
                    raise ValueError(f"Invalid H3 adaLN curve table shape {shape} in {filename}")
                if curve_grid is not None and (curve_grid, curve_dim) != shape:
                    raise ValueError("H3 checkpoint shards disagree about the adaLN curve table shape")
                curve_grid, curve_dim = shape

            reader_keys = reader.keys()
            for marker_key in (key for key in reader_keys if key.endswith(".comfy_quant")):
                module_name = marker_key[: -len(".comfy_quant")]
                try:
                    marker = reader.get_tensor(marker_key, device=torch.device("cpu"))
                    config = json.loads(marker.numpy().tobytes())
                except Exception as exc:
                    raise ValueError(f"Invalid Comfy quantization marker {marker_key} in {filename}") from exc

                params = config.get("params", {})
                if not isinstance(params, dict):
                    params = {}
                quant_format = config.get("format")
                convrot = config.get("convrot", params.get("convrot", False))
                if quant_format != "int8_tensorwise" or not convrot:
                    raise ValueError(
                        f"MiniMax H3 training supports only INT8 ConvRot quantized layers; {module_name} uses {quant_format!r}"
                    )

                weight_key = f"{module_name}.weight"
                scale_key = f"{module_name}.weight_scale"
                if weight_key not in reader.header or scale_key not in reader.header:
                    raise ValueError(f"INT8 ConvRot layer {module_name} is missing its weight or weight_scale")
                if reader.header[weight_key]["dtype"] != "I8":
                    raise ValueError(f"INT8 ConvRot layer {module_name} has non-I8 storage")
                weight_shape = tuple(reader.header[weight_key]["shape"])
                if len(weight_shape) != 2:
                    raise ValueError(f"INT8 ConvRot layer {module_name} must have a 2-D weight")
                scale_shape = tuple(reader.header[scale_key]["shape"])
                group_size = int(config.get("convrot_groupsize", params.get("convrot_groupsize", 256)))
                layer_config = Int8ConvRotConfig(group_size=group_size, scale_shape=scale_shape)
                if module_name in int8_layers and int8_layers[module_name] != layer_config:
                    raise ValueError(f"Conflicting INT8 ConvRot metadata for {module_name}")
                int8_layers[module_name] = layer_config

    return TransformerCheckpointLayout(int8_layers, curve_grid, curve_dim)


def load_selected_weights(
    model: torch.nn.Module,
    files: Iterable[Path],
    *,
    device: torch.device | str,
    dtype: torch.dtype | None | Callable[[str], torch.dtype | None],
    key_transform: Callable[[str], str | None] = lambda key: key,
    disable_numpy_memmap: bool = False,
) -> None:
    """Assign selected tensors into a meta-initialized module, one shard at a time."""

    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    for filename in files:
        shard = {}
        with MemoryEfficientSafeOpen(str(filename), disable_numpy_memmap=disable_numpy_memmap) as reader:
            reader_keys = reader.keys()
            for source_key in reader_keys:
                key = key_transform(source_key)
                if key is None or key not in expected:
                    continue
                source_dtype = reader.header[source_key]["dtype"]
                target_dtype = dtype(key) if callable(dtype) else dtype
                is_supported_int8 = source_dtype == "I8" and target_dtype == torch.int8
                if source_dtype not in {"F16", "BF16", "F32", "F64"} and not is_supported_int8:
                    raise ValueError(f"Quantized checkpoint tensor {source_key} ({source_dtype}) is not supported for H3 training")
                tensor = reader.get_tensor(source_key, device=torch.device(device), dtype=target_dtype)
                shard[key] = tensor
                loaded.add(key)
        if shard:
            model.load_state_dict(shard, strict=False, assign=True)

    missing = sorted(expected - loaded)
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"H3 checkpoint is missing {len(missing)} required tensors: {preview}")


def load_transformer(
    path: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = torch.bfloat16,
    attn_mode: str = "torch",
    split_attn: bool = False,
    disable_numpy_memmap: bool = False,
) -> MiniMaxH3Model:
    files = resolve_safetensor_files(path, "transformer")
    layout = inspect_transformer_checkpoint(files)
    with torch.device("meta"):
        model = MiniMaxH3Model(
            time_embed_dim=layout.adaln_curve_dim or 2688,
            adaln_curve_grid=layout.adaln_curve_grid,
            attn_mode=attn_mode,
            split_attn=split_attn,
        )
    if layout.int8_convrot_layers:
        apply_int8_convrot_monkey_patch(model, layout.int8_convrot_layers)
        logger.info(
            "Loading MiniMax H3%s INT8 ConvRot transformer",
            " pruned" if layout.adaln_curve_grid is not None else "",
        )
    fp32_prefixes = (
        "audio_patch_proj.",
        "video_patch_proj.",
        "time_embedder.",
        "final_layer.audio_out.",
        "final_layer.video_out.",
        "rope.inv_freq",
    )
    int8_weight_keys = {f"{name}.weight" for name in layout.int8_convrot_layers}
    int8_scale_keys = {f"{name}.weight_scale" for name in layout.int8_convrot_layers}

    def target_dtype(key: str) -> torch.dtype | None:
        if key in int8_weight_keys:
            return torch.int8
        if key in int8_scale_keys:
            return torch.float32
        if layout.adaln_curve_grid is not None and (key == "adaln_t_table" or ".adaln_proj.linear." in key):
            return torch.float32
        return torch.float32 if key.startswith(fp32_prefixes) else dtype

    load_selected_weights(
        model,
        files,
        device=device,
        dtype=target_dtype,
        disable_numpy_memmap=disable_numpy_memmap,
    )
    model.eval()
    return model
