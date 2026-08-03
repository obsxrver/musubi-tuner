import json

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.bucket import BucketSelector
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
    pack_audio,
    patchify_video,
    time_shift_sigma,
    time_shift_slope,
    unpack_audio,
    unpatchify_video,
)
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
