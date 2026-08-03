"""Encoder-only MiniMax H3 stereo audio VAE and media extraction helpers."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.minimax_h3.minimax_h3_utils import AUDIO_SAMPLE_RATE, MemoryEfficientSafeOpen, resolve_safetensor_files


LATENTS_MEAN = (
    -0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373,
    0.08179686214561671, -0.35782641352446604, 0.040623809960919084, -0.01552534501956604,
    -0.223362481667332, 0.1821006842509091, 0.2941778783780663, -0.07901167601970885,
    -0.056815072777201, -0.3699028221860095, -0.31616315591624855, 0.5905951377425391,
    -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
    -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344,
    0.02868032184767538, -0.21217779266454084, -0.1678263169941987, 0.3221287889040614,
    -0.1223055851554907, 0.4356604928128464, -0.0502599202236253, 0.3979258376211797,
)
LATENTS_STD = (
    1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828,
    1.6390226546605453, 2.7788298348882177, 1.7659090095747236, 1.6199757612137327,
    2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178,
    1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373,
    1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
    1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
    1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855,
    1.718045989838149, 1.6307219190837705, 1.8661226051202384, 1.5613768203168363,
)


def snake(x, alpha):
    return x + torch.sin(alpha * x).square() / (alpha + 1e-9)


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        pad = 3 * dilation
        self.block = nn.Sequential(
            Snake1d(dim), nn.Conv1d(dim, dim, 7, dilation=dilation, padding=pad),
            Snake1d(dim), nn.Conv1d(dim, dim, 1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim, stride):
        super().__init__()
        half = dim // 2
        self.block = nn.Sequential(
            ResidualUnit(half, 1), ResidualUnit(half, 3), ResidualUnit(half, 9), Snake1d(half),
            nn.Conv1d(half, dim, 2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model=64, strides=(2, 4, 4, 5, 5), d_latent=2048):
        super().__init__()
        blocks = [nn.Conv1d(1, d_model, 7, padding=3)]
        for stride in strides:
            d_model *= 2
            blocks.append(EncoderBlock(d_model, stride))
        blocks.extend([Snake1d(d_model), nn.Conv1d(d_model, d_latent, 3, padding=1)])
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class GeGluMlp(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(features, hidden)
        self.w1 = nn.Linear(features, hidden)
        self.w2 = nn.Linear(hidden, features)

    def forward(self, x):
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)) * self.w1(x))


class CausalAttention(nn.Module):
    def __init__(self, in_dim=2048, out_dim=32, num_heads=8):
        super().__init__()
        self.head_dim = in_dim // num_heads
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.empty(in_dim))
        self.v_bias = nn.Parameter(torch.empty(in_dim))
        self.register_buffer("zero_k_bias", torch.empty(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        b, n, _ = x.shape
        bias = torch.cat([self.q_bias, self.zero_k_bias, self.v_bias])
        q, k, v = F.linear(x, self.qkv.weight, bias).reshape(
            b, n, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = F.adaptive_avg_pool1d(x.mean(dim=1), self.out_dim)
        return self.proj(x)


class AttnProjection(nn.Module):
    def __init__(self, in_dim=2048, out_dim=32, num_heads=8, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(out_dim, int(out_dim * mlp_ratio))

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MiniMaxH3AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.samples_per_latent = 800
        self.latents_per_second = 40
        self.encoder = Encoder()
        self.pre_block = AttnProjection()
        self.mean_proj = nn.Conv1d(32, 32, 1)
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN), persistent=False)
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD), persistent=False)

    @property
    def device(self):
        return self.mean_proj.weight.device

    @property
    def dtype(self):
        return self.mean_proj.weight.dtype

    def encode(self, waveform, target_latent_length: Optional[int] = None):
        b, stereo, length = waveform.shape
        right_pad = math.ceil(length / self.samples_per_latent) * self.samples_per_latent - length
        x = F.pad(waveform, (0, right_pad)).reshape(b * stereo, 1, -1)
        x = self.encoder(x)
        x = self.pre_block(x.transpose(1, 2)).transpose(1, 2)
        z = self.mean_proj(x)
        z = (z - self.latents_mean.view(1, -1, 1).to(z)) / self.latents_std.view(1, -1, 1).to(z)
        z = z.reshape(b, stereo, 32, -1).permute(0, 2, 1, 3)
        if target_latent_length is not None:
            if z.shape[-1] < target_latent_length:
                z = F.pad(z, (0, target_latent_length - z.shape[-1]))
            else:
                z = z[..., :target_latent_length]
        return z


def _canonical_audio_key(key: str) -> str:
    for prefix in ("model.", "audio_vae."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def _weight_norm_base(key: str):
    suffixes = {
        ".weight_g": "g", ".weight_v": "v",
        ".parametrizations.weight.original0": "g", ".parametrizations.weight.original1": "v",
    }
    for suffix, part in suffixes.items():
        if key.endswith(suffix):
            return key[: -len(suffix)] + ".weight", part
    return None, None


def load_audio_vae(
    path: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    disable_numpy_memmap: bool = False,
) -> MiniMaxH3AudioEncoder:
    with torch.device("meta"):
        model = MiniMaxH3AudioEncoder()
    expected = set(model.state_dict())
    state = {}
    norm_parts = {}
    for filename in resolve_safetensor_files(path, "audio_vae"):
        with MemoryEfficientSafeOpen(str(filename), disable_numpy_memmap=disable_numpy_memmap) as reader:
            for source_key in reader.keys():
                key = _canonical_audio_key(source_key)
                if key in expected:
                    if reader.header[source_key]["dtype"] not in {"F16", "BF16", "F32", "F64"}:
                        raise ValueError("Quantized H3 audio VAE weights are not supported")
                    state[key] = reader.get_tensor(source_key, device=torch.device(device), dtype=dtype)
                    continue
                base, part = _weight_norm_base(key)
                if base in expected:
                    if reader.header[source_key]["dtype"] not in {"F16", "BF16", "F32", "F64"}:
                        raise ValueError("Quantized H3 audio VAE weights are not supported")
                    norm_parts.setdefault(base, {})[part] = reader.get_tensor(
                        source_key, device=torch.device(device), dtype=dtype
                    )
    for key, parts in norm_parts.items():
        if set(parts) != {"g", "v"}:
            continue
        g, v = parts["g"], parts["v"]
        dims = tuple(range(1, v.ndim))
        state[key] = v * (g / torch.linalg.vector_norm(v.float(), dim=dims, keepdim=True).to(v.dtype).clamp_min(1e-12))
    missing = sorted(expected - set(state))
    if missing:
        raise ValueError(f"H3 audio VAE checkpoint is missing {len(missing)} encoder tensors: {', '.join(missing[:8])}")
    model.load_state_dict(state, strict=True, assign=True)
    model.latents_mean = torch.tensor(LATENTS_MEAN, device=device)
    model.latents_std = torch.tensor(LATENTS_STD, device=device)
    model.eval().requires_grad_(False)
    return model


def load_audio_segment(path: str, start_frame: int, frame_count: int, fps: int = 24) -> torch.Tensor:
    """Decode, stereo-resample, and crop the audio corresponding to a cached video clip."""

    sample_count = round(frame_count * AUDIO_SAMPLE_RATE / fps)
    start_sample = round(start_frame * AUDIO_SAMPLE_RATE / fps)
    try:
        import av

        pieces = []
        with av.open(path) as container:
            if not container.streams.audio:
                return torch.zeros(2, sample_count)
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=AUDIO_SAMPLE_RATE)
            for frame in container.decode(audio=0):
                converted = resampler.resample(frame)
                converted = converted if isinstance(converted, list) else [converted]
                pieces.extend(item.to_ndarray() for item in converted if item is not None)
            flushed = resampler.resample(None)
            flushed = flushed if isinstance(flushed, list) else [flushed]
            pieces.extend(item.to_ndarray() for item in flushed if item is not None)
        if not pieces:
            return torch.zeros(2, sample_count)
        waveform = np.concatenate(pieces, axis=-1)
        waveform = torch.from_numpy(waveform).float()
        waveform = waveform[:, start_sample : start_sample + sample_count]
        if waveform.shape[-1] < sample_count:
            waveform = F.pad(waveform, (0, sample_count - waveform.shape[-1]))
        return waveform.clamp_(-1.0, 1.0)
    except (av.error.FFmpegError, OSError, ValueError):
        return torch.zeros(2, sample_count)
