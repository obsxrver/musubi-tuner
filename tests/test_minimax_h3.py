import torch

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.minimax_h3.minimax_h3_utils import (
    align_frame_count,
    audio_latent_length,
    is_valid_frame_count,
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
    numerical = (time_shift_sigma(sigma + epsilon, 12.0, 3.0) - time_shift_sigma(sigma - epsilon, 12.0, 3.0)) / (
        2 * epsilon
    )
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
