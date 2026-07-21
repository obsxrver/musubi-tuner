from unittest.mock import patch

import torch

from musubi_tuner.wan.modules.model import WanModel
from musubi_tuner.wan.modules.vae import WanVAE, WanVAE_


def test_wan_2_2_scalar_timestep_embedding_matches_per_token_embedding():
    torch.manual_seed(0)
    model = WanModel(
        model_type="t2v",
        model_version="2.2",
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=20,
        out_dim=4,
        num_heads=4,
        num_layers=0,
        attn_mode="torch",
    )
    timesteps = torch.tensor([123.0, 456.0])
    seq_len = 8

    scalar_e, scalar_e0 = model.embed_timesteps(timesteps, seq_len)
    token_e, token_e0 = model.embed_timesteps(timesteps[:, None].expand(-1, seq_len).clone(), seq_len)

    assert scalar_e.shape == (2, 1, model.dim)
    assert scalar_e0.shape == (2, 1, 6, model.dim)
    assert token_e.shape == (2, seq_len, model.dim)
    assert token_e0.shape == (2, seq_len, 6, model.dim)
    assert torch.allclose(scalar_e.expand_as(token_e), token_e, atol=1e-6, rtol=1e-5)
    assert torch.allclose(scalar_e0.expand_as(token_e0), token_e0, atol=1e-6, rtol=1e-5)


class _FakeBatchedVAEModel:
    def __init__(self):
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, videos, scale):
        self.encode_calls += 1
        return videos + 1.0

    def decode(self, latents, scale):
        self.decode_calls += 1
        return latents * 0.5


def test_wan_vae_wrapper_batches_tensor_inputs():
    vae = WanVAE.__new__(WanVAE)
    vae.model = _FakeBatchedVAEModel()
    vae.scale = [0.0, 1.0]
    videos = torch.zeros(3, 2, 1, 4, 4)

    encoded = vae.encode(videos)
    decoded = vae.decode(torch.ones_like(videos))

    assert vae.model.encode_calls == 1
    assert vae.model.decode_calls == 1
    assert len(encoded) == len(decoded) == videos.shape[0]
    assert torch.equal(torch.stack(encoded), videos + 1.0)
    assert torch.equal(torch.stack(decoded), torch.full_like(videos, 0.5))


class _FakeChunkedVAE:
    z_dim = 1
    _enc_feat_map = []
    _feat_map = []

    def clear_cache(self):
        self._enc_feat_map = []
        self._feat_map = []

    def encoder(self, x, feat_cache, feat_idx):
        return x

    def decoder(self, x, feat_cache, feat_idx):
        return x

    def conv1(self, x):
        return x.repeat(1, 2, 1, 1, 1)

    def conv2(self, x):
        return x


def test_wan_vae_chunk_outputs_are_concatenated_once():
    vae = _FakeChunkedVAE()
    video = torch.arange(9.0).view(1, 1, 9, 1, 1)
    original_cat = torch.cat

    with patch("musubi_tuner.wan.modules.vae.torch.cat", wraps=original_cat) as cat:
        encoded = WanVAE_.encode(vae, video, [0.0, 1.0])
    assert cat.call_count == 1
    assert torch.equal(encoded, video)

    with patch("musubi_tuner.wan.modules.vae.torch.cat", wraps=original_cat) as cat:
        decoded = WanVAE_.decode(vae, video, [0.0, 1.0])
    assert cat.call_count == 1
    assert torch.equal(decoded, video)
