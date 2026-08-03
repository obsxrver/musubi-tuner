"""Qwen3-VL layer-50 text features used by MiniMax H3 FL2VA."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from musubi_tuner.minimax_h3.minimax_h3_utils import load_selected_weights, resolve_safetensor_files


def _text_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    return Qwen3VLTextConfig(
        vocab_size=151936,
        hidden_size=5120,
        intermediate_size=25600,
        num_hidden_layers=50,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000,
        rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [24, 20, 20]},
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        bos_token_id=151643,
        eos_token_id=151645,
    )


def _text_key(source_key: str) -> Optional[str]:
    key = source_key
    prefixes = (
        "text_encoders.qwen3vl_32b.transformer.",
        "qwen3vl_32b.transformer.",
        "model.language_model.",
        "language_model.",
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    if key.startswith("model.language_model."):
        key = key[len("model.language_model.") :]
    elif key.startswith("model."):
        key = key[len("model.") :]
    if key == "norm.weight" or key.startswith("layers.50."):
        return None
    if key.startswith("layers."):
        try:
            if int(key.split(".", 2)[1]) >= 50:
                return None
        except (ValueError, IndexError):
            pass
    return key


def load_text_encoder(
    path: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    disable_numpy_memmap: bool = False,
):
    """Load only layers 0..49, returning the unnormalized layer-50 hidden state."""

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    with torch.device("meta"):
        model = Qwen3VLTextModel(_text_config())
        model.norm = nn.Identity()
    load_selected_weights(
        model,
        resolve_safetensor_files(path, "text_encoder"),
        device=device,
        dtype=dtype,
        key_transform=_text_key,
        disable_numpy_memmap=disable_numpy_memmap,
    )
    model.eval().requires_grad_(False)
    return model


def load_tokenizer(path: Optional[str], text_encoder_path: Optional[str] = None):
    from transformers import AutoTokenizer

    if path is not None:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if text_encoder_path:
        candidate = Path(text_encoder_path)
        search = [candidate, candidate.parent] if candidate.is_dir() else [candidate.parent]
        for root in search:
            if (root / "tokenizer.json").exists():
                return AutoTokenizer.from_pretrained(root, trust_remote_code=True)
            if (root / "FL2VA" / "tokenizer" / "tokenizer.json").exists():
                return AutoTokenizer.from_pretrained(root / "FL2VA" / "tokenizer", trust_remote_code=True)
    return AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-H3", subfolder="FL2VA/tokenizer", trust_remote_code=True)


@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompts: list[str], device, max_length: int = 1024):
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    # H3 represents an otherwise empty prompt by Qwen3-VL's BOS token.
    if encoded.input_ids.shape[1] == 0:
        input_ids = torch.full((len(prompts), 1), 151643, device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        empty_rows = attention_mask.sum(dim=1) == 0
        if empty_rows.any():
            input_ids[empty_rows, 0] = 151643
            attention_mask[empty_rows, 0] = 1
    hidden = text_encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    outputs = []
    tags = []
    for index in range(hidden.shape[0]):
        length = int(attention_mask[index].sum())
        outputs.append(hidden[index, :length].contiguous())
        tags.append(torch.ones(length, device=hidden.device, dtype=torch.int64))
    return outputs, tags
