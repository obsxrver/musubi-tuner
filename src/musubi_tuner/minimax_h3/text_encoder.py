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


def _multimodal_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config=_text_config().to_dict(),
        vision_config={
            "depth": 27,
            "hidden_size": 1152,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 4304,
            "num_heads": 16,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 5120,
            "num_position_embeddings": 2304,
            "deepstack_visual_indexes": [8, 16, 24],
        },
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


def _multimodal_key(source_key: str) -> Optional[str]:
    """Map official and Comfy Qwen3-VL keys into a truncated Qwen3VLModel."""
    key = source_key
    prefixes = (
        "text_encoders.qwen3vl_32b.transformer.",
        "qwen3vl_32b.transformer.",
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break

    if key.startswith("model.language_model."):
        key = "language_model." + key[len("model.language_model.") :]
    elif key.startswith("language_model."):
        pass
    elif key.startswith("model.visual."):
        key = "visual." + key[len("model.visual.") :]
    elif key.startswith("visual."):
        pass
    elif key.startswith("model."):
        # Comfy's converted checkpoint calls the text tower `model`.
        key = "language_model." + key[len("model.") :]
    else:
        return None

    if key == "language_model.norm.weight" or key.startswith("language_model.layers.50."):
        return None
    if key.startswith("language_model.layers."):
        try:
            if int(key.split(".", 3)[2]) >= 50:
                return None
        except (ValueError, IndexError):
            pass
    return key


def load_text_encoder(
    path: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    include_visual: bool = False,
    disable_numpy_memmap: bool = False,
):
    """Load only layers 0..49, returning the unnormalized layer-50 hidden state."""

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextModel

    with torch.device("meta"):
        if include_visual:
            model = Qwen3VLModel(_multimodal_config())
            model.language_model.norm = nn.Identity()
        else:
            model = Qwen3VLTextModel(_text_config())
            model.norm = nn.Identity()
    load_selected_weights(
        model,
        resolve_safetensor_files(path, "text_encoder"),
        device=device,
        dtype=dtype,
        key_transform=_multimodal_key if include_visual else _text_key,
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


def load_processor(path: Optional[str], text_encoder_path: Optional[str] = None):
    from transformers import AutoProcessor

    if path is not None:
        return AutoProcessor.from_pretrained(path, trust_remote_code=True)
    if text_encoder_path:
        candidate = Path(text_encoder_path)
        search = [candidate, candidate.parent] if candidate.is_dir() else [candidate.parent]
        for root in search:
            for processor_dir in (root / "FL2VA" / "processor", root / "processor", root / "FL2VA" / "text_encoder"):
                if (processor_dir / "preprocessor_config.json").exists():
                    return AutoProcessor.from_pretrained(processor_dir, trust_remote_code=True)
    return AutoProcessor.from_pretrained("MiniMaxAI/MiniMax-H3", subfolder="FL2VA/processor", trust_remote_code=True)


@torch.no_grad()
def encode_prompts(
    tokenizer,
    text_encoder,
    prompts: list[str],
    device,
    max_length: int = 1024,
    processor=None,
    images: Optional[list] = None,
):
    if images is not None:
        if processor is None or len(images) != len(prompts):
            raise ValueError("H3 I2V text encoding requires one processor image per prompt")
        outputs = []
        tags = []
        for prompt, image in zip(prompts, images):
            vision = processor.image_processor(images=[image], return_tensors="pt")
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
            merge_size = processor.image_processor.merge_size**2
            num_image_tokens = int(image_grid_thw[0].prod()) // merge_size

            label_ids = tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
            vision_ids = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            input_ids = torch.tensor([label_ids + vision_ids + prompt_ids], device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            hidden = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(device=device, dtype=text_encoder.dtype),
                image_grid_thw=image_grid_thw.to(device),
                use_cache=False,
            ).last_hidden_state[0]
            outputs.append(hidden.contiguous())
            tags.append(
                torch.tensor(
                    [1] * len(label_ids) + [0] * len(vision_ids) + [1] * len(prompt_ids),
                    device=hidden.device,
                    dtype=torch.int64,
                )
            )
        return outputs, tags

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
