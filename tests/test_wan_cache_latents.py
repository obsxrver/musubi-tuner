from types import SimpleNamespace
from unittest.mock import patch

import torch

from musubi_tuner import wan_cache_latents


class _FakeWanVAE:
    device = torch.device("cpu")
    dtype = torch.bfloat16

    def __init__(self):
        self.batch_sizes = []

    def encode_tensor(self, videos):
        self.batch_sizes.append(videos.shape[0])
        return torch.stack([torch.full((2, 1, 1, 1), float(i + 1)) for i in range(videos.shape[0])])


class _FakeClip:
    device = torch.device("cpu")

    def __init__(self):
        self.input_shapes = []

    def visual(self, videos):
        self.input_shapes.append(tuple(videos.shape))
        batch_frames = videos.shape[0] * videos.shape[2]
        return torch.arange(batch_frames * 6, dtype=torch.float32).reshape(batch_frames, 2, 3)


def test_one_frame_cache_batches_vae_frames_and_clip_images():
    contents = torch.zeros(2, 3, 2, 4, 4)
    items = [
        SimpleNamespace(
            item_key=f"item-{i}",
            latent_cache_path=f"item-{i}.safetensors",
            fp_1f_clean_indices=[0],
            fp_1f_target_index=1,
        )
        for i in range(2)
    ]
    vae = _FakeWanVAE()
    clip = _FakeClip()
    saved = []
    wan_cache_latents.black_image_latents.clear()

    with (
        patch.object(wan_cache_latents.cache_latents, "preprocess_contents", return_value=(None, None, contents, None)),
        patch.object(wan_cache_latents, "save_latent_cache_wan", side_effect=lambda *args, **kwargs: saved.append((args, kwargs))),
    ):
        wan_cache_latents.encode_and_save_batch_one_frame(vae, clip, items)

    assert vae.batch_sizes == [4, 1]
    assert clip.input_shapes == [(2, 3, 1, 4, 4)]
    assert len(saved) == 2
    for args, kwargs in saved:
        _, latent, clip_context, image_latent, control_latent = args
        assert latent.shape == (2, 2, 1, 1)
        assert clip_context.shape == (1, 2, 3)
        assert image_latent.shape == (6, 2, 1, 1)
        assert control_latent is None
        assert kwargs["f_indices"] == [0, 1]
