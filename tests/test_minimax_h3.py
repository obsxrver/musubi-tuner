import json
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.bucket import BucketBatchManager, BucketSelector
from musubi_tuner.dataset.cache_io import (
    save_latent_cache_minimax_h3,
    save_text_encoder_output_cache_minimax_h3,
)
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.minimax_h3_utils import (
    align_frame_count,
    audio_latent_length,
    inspect_transformer_checkpoint,
    is_valid_frame_count,
    load_selected_weights,
    video_latent_length,
)
from musubi_tuner.minimax_h3.model import (
    MiniMaxH3Model,
    PackedLayout,
    pack_audio,
    patchify_video,
    time_shift_sigma,
    time_shift_slope,
    unpack_audio,
    unpatchify_video,
)
from musubi_tuner.minimax_h3.text_encoder import _multimodal_key, encode_prompts
from musubi_tuner.minimax_h3.video_vae import MiniMaxH3VideoEncoder
from musubi_tuner.modules.int8_optimization_utils import (
    Int8ConvRotConfig,
    _convrot_hadamard,
    apply_int8_convrot_monkey_patch,
)


def test_h3_frame_and_audio_alignment():
    assert align_frame_count(5) == 5
    assert align_frame_count(21) == 5
    assert align_frame_count(22) == 22
    assert align_frame_count(23, "up") == 39
    assert is_valid_frame_count(39)
    assert not is_valid_frame_count(38)
    assert video_latent_length(5) == 2
    assert video_latent_length(22) == 7
    assert video_latent_length(39) == 12
    assert audio_latent_length(24) == 40
    assert BucketSelector((512, 512), architecture=ARCHITECTURE_MINIMAX_H3).reso_steps == 32


def test_h3_time_shift_slope_matches_derivative():
    sigma = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
    epsilon = 1e-6
    numerical = (time_shift_sigma(sigma + epsilon, 12.0, 3.0) - time_shift_sigma(sigma - epsilon, 12.0, 3.0)) / (2 * epsilon)
    torch.testing.assert_close(time_shift_slope(sigma, 12.0, 3.0), numerical, rtol=1e-6, atol=1e-7)


def test_h3_pack_round_trips():
    video = torch.randn(2, 3, 2, 6, 8)
    rows = patchify_video(video)
    restored = unpatchify_video(rows, 2, 3, 4, 3)
    torch.testing.assert_close(restored, video)

    audio = torch.randn(2, 4, 2, 7)
    torch.testing.assert_close(unpack_audio(pack_audio(audio)), audio)


def test_h3_first_frame_layout_and_checkpoint_key_mapping():
    layout = PackedLayout(text_len=4, latent_t=2, latent_h=4, latent_w=4, audio_t=3, condition_t=1)
    assert layout.text_slice == (0, 4)
    assert layout.condition_slice == (4, 8)
    assert layout.audio_slice == (8, 14)
    assert layout.video_slice == (14, 22)
    torch.testing.assert_close(layout.position_ids[layout.condition_slice[0], 0], torch.tensor(4.0, dtype=torch.float64))

    assert _multimodal_key("model.visual.patch_embed.proj.weight") == "visual.patch_embed.proj.weight"
    assert _multimodal_key("model.layers.0.self_attn.q_proj.weight") == "language_model.layers.0.self_attn.q_proj.weight"
    assert _multimodal_key("model.language_model.layers.49.mlp.down_proj.weight") == (
        "language_model.layers.49.mlp.down_proj.weight"
    )
    assert _multimodal_key("model.language_model.layers.50.mlp.down_proj.weight") is None


def test_h3_first_frame_cache_collates_as_latents_image(tmp_path):
    item = ItemInfo(
        item_key="sample_00000-022",
        caption="test prompt",
        original_size=(32, 32),
        bucket_size=(32, 32),
        frame_count=22,
        latent_cache_path=str(tmp_path / "sample_00000-022_h3.safetensors"),
    )
    item.text_encoder_output_cache_path = str(tmp_path / "sample_00000-022_h3_te.safetensors")

    video = torch.randn(2, 7, 4, 4, dtype=torch.bfloat16)
    audio = torch.randn(3, 2, 40, dtype=torch.bfloat16)
    image = torch.randn(2, 1, 4, 4, dtype=torch.bfloat16)
    embed = torch.randn(5, 8, dtype=torch.bfloat16)
    tags = torch.tensor([1, 1, 0, 0, 1], dtype=torch.long)
    save_latent_cache_minimax_h3(item, video, audio, image)
    save_text_encoder_output_cache_minimax_h3(item, embed, tags)

    batch = BucketBatchManager({(32, 32, 22): [item]}, batch_size=1)[0]
    torch.testing.assert_close(batch["latents"], video.unsqueeze(0))
    torch.testing.assert_close(batch["latents_audio"], audio.unsqueeze(0))
    torch.testing.assert_close(batch["latents_image"], image.unsqueeze(0))
    assert len(batch["h3_text_embed"]) == len(batch["h3_token_tags"]) == 1
    torch.testing.assert_close(batch["h3_text_embed"][0], embed)
    torch.testing.assert_close(batch["h3_token_tags"][0], tags)


def test_h3_first_frame_text_presentation_tags_vision_rows():
    class Tokenizer:
        def __call__(self, value, **kwargs):
            return {"input_ids": [10, 11] if value.startswith("<Picture") else [20, 21, 22]}

        def convert_tokens_to_ids(self, value):
            return {"<|vision_start|>": 30, "<|image_pad|>": 31, "<|vision_end|>": 32}[value]

    class ImageProcessor:
        merge_size = 2

        def __call__(self, images, return_tensors):
            return {"pixel_values": torch.zeros(4, 3), "image_grid_thw": torch.tensor([[1, 4, 4]])}

    class TextEncoder:
        dtype = torch.float32

        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.zeros(1, input_ids.shape[1], 8))

    embeds, tags = encode_prompts(
        Tokenizer(),
        TextEncoder(),
        ["prompt"],
        torch.device("cpu"),
        processor=SimpleNamespace(image_processor=ImageProcessor()),
        images=[torch.zeros(32, 32, 3).numpy()],
    )
    assert embeds[0].shape == (11, 8)
    assert tags[0].tolist() == [1, 1] + [0] * 6 + [1, 1, 1]


def test_h3_keyframe_encode_uses_seeded_sample_and_one_frame():
    class DummyEncoder:
        pixel_mean = torch.zeros(1, 3, 1, 1, 1)
        pixel_std = torch.ones(1, 3, 1, 1, 1)
        latents_mean = torch.zeros(2)
        latents_std = torch.ones(2)

        def _adaptive_encode(self, value):
            return torch.zeros(value.shape[0], 4, 1, value.shape[-2], value.shape[-1])

    pixels = torch.zeros(1, 3, 1, 2, 2)
    first = MiniMaxH3VideoEncoder.encode_keyframe(DummyEncoder(), pixels, seed=42)
    repeated = MiniMaxH3VideoEncoder.encode_keyframe(DummyEncoder(), pixels, seed=42)
    different = MiniMaxH3VideoEncoder.encode_keyframe(DummyEncoder(), pixels, seed=43)
    assert first.shape == (1, 2, 1, 2, 2)
    torch.testing.assert_close(first, repeated)
    assert not torch.equal(first, different)


def test_tiny_h3_forward_and_backward():
    model = MiniMaxH3Model(
        hidden_size=12,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=6,
        ffn_hidden_size=16,
        latents_dim=2,
        audio_latents_dim=3,
        text_dim=8,
        timestep_input_dim=8,
        time_embed_hidden_size=12,
        time_embed_dim=6,
        rope_inv_freq_len=1,
    )
    with torch.no_grad():
        model.rope.inv_freq.fill_(1.0)
    video = torch.randn(1, 2, 2, 4, 4, requires_grad=True)
    audio = torch.randn(1, 3, 2, 3, requires_grad=True)
    context = torch.randn(1, 4, 8, requires_grad=True)
    video_out, audio_out = model(video, audio, torch.tensor([0.7]), context)
    assert video_out.shape == video.shape
    assert audio_out.shape == audio.shape
    (video_out.square().mean() + audio_out.square().mean()).backward()
    assert video.grad is not None
    assert audio.grad is not None
    assert context.grad is not None


def test_tiny_h3_first_frame_forward_and_backward():
    model = MiniMaxH3Model(
        hidden_size=12,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=6,
        ffn_hidden_size=16,
        latents_dim=2,
        audio_latents_dim=3,
        text_dim=8,
        timestep_input_dim=8,
        time_embed_hidden_size=12,
        time_embed_dim=6,
        rope_inv_freq_len=1,
    )
    with torch.no_grad():
        model.rope.inv_freq.fill_(1.0)
    video = torch.randn(1, 2, 2, 4, 4, requires_grad=True)
    first_frame = torch.randn(1, 2, 1, 4, 4, requires_grad=True)
    audio = torch.randn(1, 3, 2, 3, requires_grad=True)
    context = torch.randn(1, 5, 8, requires_grad=True)
    tags = torch.tensor([[1, 1, 0, 0, 1]])
    video_out, audio_out = model(
        video,
        audio,
        torch.tensor([0.7]),
        context,
        tags,
        condition_video=first_frame,
    )
    assert video_out.shape == video.shape
    assert audio_out.shape == audio.shape
    (video_out.square().mean() + audio_out.square().mean()).backward()
    assert first_frame.grad is not None


def test_int8_convrot_linear_forward_and_backward():
    torch.manual_seed(123)
    layer = torch.nn.Linear(16, 12)
    original_weight = layer.weight.detach().clone()
    rotated_weight = _convrot_hadamard(original_weight, 16)
    weight_scale = (rotated_weight.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-30)
    quantized_weight = (rotated_weight / weight_scale).round().clamp(-128, 127).to(torch.int8)

    apply_int8_convrot_monkey_patch(layer, {"": Int8ConvRotConfig(16, tuple(weight_scale.shape))})
    layer.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    layer.weight_scale.copy_(weight_scale)
    assert layer.__class__.__name__ == "Linear"  # LoRA discovery relies on the exact class name.

    dequantized_weight = _convrot_hadamard(quantized_weight.float() * weight_scale, 16)
    value = torch.randn(5, 16, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_(True)
    probe = torch.randn(5, 12)

    output = layer(value)
    reference = F.linear(reference_value, dequantized_weight, layer.bias)
    torch.testing.assert_close(output, reference, rtol=0.04, atol=0.04)

    (output * probe).sum().backward()
    (reference * probe).sum().backward()
    torch.testing.assert_close(value.grad, reference_value.grad, rtol=1e-5, atol=1e-5)

    bf16_value = torch.randn(2, 16, dtype=torch.bfloat16, requires_grad=True)
    bf16_output = layer(bf16_value)
    assert bf16_output.dtype == torch.bfloat16
    bf16_output.float().sum().backward()
    assert bf16_value.grad is not None


def test_h3_int8_checkpoint_metadata_and_pruned_curve(tmp_path):
    marker = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}).encode()),
        dtype=torch.uint8,
    )
    checkpoint = tmp_path / "h3_int8.safetensors"
    save_file(
        {
            "fc.weight": torch.zeros(8, 64, dtype=torch.int8),
            "fc.weight_scale": torch.ones(8, 1),
            "fc.comfy_quant": marker,
            "adaln_t_table": torch.zeros(17, 4),
        },
        checkpoint,
    )

    layout = inspect_transformer_checkpoint([checkpoint])
    assert layout.adaln_curve_grid == 17
    assert layout.adaln_curve_dim == 4
    assert layout.int8_convrot_layers["fc"] == Int8ConvRotConfig(64, (8, 1))

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(64, 8, bias=False)

    with torch.device("meta"):
        model = TinyModel()
    apply_int8_convrot_monkey_patch(model, layout.int8_convrot_layers)
    load_selected_weights(
        model,
        [checkpoint],
        device="cpu",
        dtype=lambda key: torch.int8 if key == "fc.weight" else torch.float32,
    )
    assert model.fc.weight.dtype == torch.int8
    value = torch.randn(2, 64, requires_grad=True)
    model.fc(value).sum().backward()
    assert value.grad is not None


def test_tiny_pruned_h3_forward_and_backward():
    model = MiniMaxH3Model(
        hidden_size=12,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=6,
        ffn_hidden_size=16,
        latents_dim=2,
        audio_latents_dim=3,
        text_dim=8,
        time_embed_dim=4,
        adaln_curve_grid=9,
        rope_inv_freq_len=1,
    )
    with torch.no_grad():
        model.rope.inv_freq.fill_(1.0)
        model.adaln_t_table.normal_()
    video = torch.randn(1, 2, 2, 4, 4, requires_grad=True)
    audio = torch.randn(1, 3, 2, 3, requires_grad=True)
    context = torch.randn(1, 4, 8, requires_grad=True)
    video_out, audio_out = model(video, audio, torch.tensor([0.7]), context)
    (video_out.square().mean() + audio_out.square().mean()).backward()
    assert video.grad is not None
    assert audio.grad is not None
    assert context.grad is not None
