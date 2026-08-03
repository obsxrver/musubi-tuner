"""Checkpoint and shape utilities for MiniMax H3."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch

from musubi_tuner.minimax_h3.model import MiniMaxH3Model
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


VIDEO_FPS = 24
AUDIO_SAMPLE_RATE = 32_000
AUDIO_LATENTS_PER_SECOND = 40
VIDEO_FRAME_STRIDE = 17
VIDEO_FRAME_OFFSET = 5


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


def resolve_safetensor_files(path: str, component: Optional[str] = None) -> list[Path]:
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


def load_selected_weights(
    model: torch.nn.Module,
    files: Iterable[Path],
    *,
    device: torch.device | str,
    dtype: Optional[torch.dtype] | Callable[[str], Optional[torch.dtype]],
    key_transform: Callable[[str], Optional[str]] = lambda key: key,
    disable_numpy_memmap: bool = False,
) -> None:
    """Assign selected tensors into a meta-initialized module, one shard at a time."""

    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    for filename in files:
        shard = {}
        with MemoryEfficientSafeOpen(str(filename), disable_numpy_memmap=disable_numpy_memmap) as reader:
            for source_key in reader.keys():
                key = key_transform(source_key)
                if key is None or key not in expected:
                    continue
                source_dtype = reader.header[source_key]["dtype"]
                if source_dtype not in {"F16", "BF16", "F32", "F64"}:
                    raise ValueError(
                        f"Quantized checkpoint tensor {source_key} ({source_dtype}) is not supported for H3 training; use BF16 weights"
                    )
                target_dtype = dtype(key) if callable(dtype) else dtype
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
    dtype: Optional[torch.dtype] = torch.bfloat16,
    attn_mode: str = "torch",
    split_attn: bool = False,
    disable_numpy_memmap: bool = False,
) -> MiniMaxH3Model:
    files = resolve_safetensor_files(path, "transformer")
    with torch.device("meta"):
        model = MiniMaxH3Model(attn_mode=attn_mode, split_attn=split_attn)
    fp32_prefixes = (
        "audio_patch_proj.",
        "video_patch_proj.",
        "time_embedder.",
        "final_layer.audio_out.",
        "final_layer.video_out.",
        "rope.inv_freq",
    )
    load_selected_weights(
        model,
        files,
        device=device,
        dtype=lambda key: torch.float32 if key.startswith(fp32_prefixes) else dtype,
        disable_numpy_memmap=disable_numpy_memmap,
    )
    model.eval()
    return model
