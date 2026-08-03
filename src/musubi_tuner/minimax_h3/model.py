"""MiniMax H3 joint audio/video diffusion transformer.

H3 packs text, stereo-audio, and video tokens into one full-attention stream.
The checkpoint predicts ``clean - noise`` for each target stream.  Video and
audio share a base flow time but use different rational sigma shifts.

The parameter names intentionally follow the official and ComfyUI checkpoint
layouts so either BF16 distribution can be loaded without a key conversion.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from musubi_tuner.modules.attention import AttentionParams
from musubi_tuner.modules.attention import attention as common_attention
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VIDEO_SIGMA_SHIFT = 12.0
AUDIO_SIGMA_SHIFT = 3.0


def time_shift_sigma(sigma: torch.Tensor, from_shift: float, to_shift: float) -> torch.Tensor:
    """Map a shifted flow sigma to another shift on the same base grid."""

    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma: torch.Tensor, from_shift: float, to_shift: float) -> torch.Tensor:
    """Derivative of :func:`time_shift_sigma` with respect to ``sigma``."""

    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    numerator = to_shift * (1.0 + (from_shift - 1.0) * base) ** 2
    denominator = from_shift * (1.0 + (to_shift - 1.0) * base) ** 2
    return numerator / denominator


def patchify_video(latent: torch.Tensor, patch_size: tuple[int, int, int] = (1, 2, 2)) -> torch.Tensor:
    """Convert ``[B,C,T,H,W]`` video latents to rows of flattened patches."""

    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    if t_full % pt or h_full % ph or w_full % pw:
        raise ValueError(f"latent shape {tuple(latent.shape)} is not divisible by patch size {patch_size}")
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b, t * h * w, c * pt * ph * pw)


def unpatchify_video(
    rows: torch.Tensor,
    t: int,
    h: int,
    w: int,
    channels: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    """Inverse of :func:`patchify_video`."""

    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, channels, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, channels, t * pt, h * ph, w * pw)


def pack_audio(latent: torch.Tensor) -> torch.Tensor:
    """Pack ``[B,C,stereo,T]`` latents in channel-major stereo order."""

    b, c, stereo, t = latent.shape
    return latent.permute(0, 2, 3, 1).reshape(b, stereo * t, c)


def unpack_audio(rows: torch.Tensor, stereo: int = 2) -> torch.Tensor:
    """Inverse of :func:`pack_audio`."""

    b, sequence, channels = rows.shape
    t = sequence // stereo
    return rows.reshape(b, stereo, t, channels).permute(0, 3, 1, 2)


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    area = math.sqrt(height * width)
    h_axis = _axis_from_sqrt_area(height, 2, area)
    w_axis = _axis_from_sqrt_area(width, 2, area)
    hh, ww = torch.meshgrid(h_axis, w_axis, indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_axis


def _video_t_spans(count: int) -> list[float]:
    return [FRAME_RESCALE * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)] for index in range(count)]


def _video_t_grid(count: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(_video_t_spans(count), dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _audio_grid(cursor: float, length: int, width_low: float, width_high: float) -> torch.Tensor:
    grid = torch.zeros(length * 2, 3, dtype=torch.float64)
    grid[:, 0] = (cursor + torch.arange(length, dtype=torch.float64)).repeat(2)
    grid[:length, 2] = width_low
    grid[length:, 2] = width_high
    return grid


def _video_grid(length: int, frame_grid: torch.Tensor, cursor: float) -> torch.Tensor:
    grid = torch.empty(length, frame_grid.shape[0], 3, dtype=torch.float64)
    grid[:, :, 0] = _video_t_grid(length, cursor)[:, None]
    grid[:, :, 1:] = frame_grid[None]
    return grid.reshape(-1, 3)


def apply_split_half_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply H3's split-half partial RoPE to ``[B,S,H,D]`` tensors."""

    half = angles.shape[-1] // 2
    rot = half * 2
    angle = angles[..., :half].to(device=x.device, dtype=torch.float32)
    cos = angle.cos().to(x.dtype)[None, :, None, :]
    sin = angle.sin().to(x.dtype)[None, :, None, :]
    first, second = x[..., :half], x[..., half:rot]
    rotated = torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)
    return torch.cat([rotated, x[..., rot:]], dim=-1)


class PackedLayout:
    """Packed ``[text | audio | video]`` layout for T2VA training."""

    def __init__(self, text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int):
        frame, width_grid = _frame_grid(latent_h, latent_w)
        cursor = float(text_len)

        text_pos = torch.zeros(text_len, 3, dtype=torch.float64)
        text_pos[:, 0] = torch.arange(text_len, dtype=torch.float64)
        audio_pos = _audio_grid(cursor, audio_t, float(width_grid[0]), float(width_grid[-1]))
        video_pos = _video_grid(latent_t, frame, cursor)

        audio_start = text_len
        video_start = audio_start + audio_t * 2
        self.text_slice = (0, text_len)
        self.audio_slice = (audio_start, video_start)
        self.video_slice = (video_start, video_start + video_pos.shape[0])
        self.position_ids = torch.cat([text_pos, audio_pos, video_pos], dim=0)
        self.seq_len = self.position_ids.shape[0]
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)


class TimeEmbedder(nn.Module):
    def __init__(self, freq_dim: int, hidden: int, out: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden, bias=True, dtype=torch.float32)
        self.proj_out = nn.Linear(hidden, out, bias=True, dtype=torch.float32)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half)
        args = timestep.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj_out(F.silu(self.proj_in(embedding)))


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, eps: float, attn_mode: str, split_attn: bool):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.attn_params = AttentionParams.create_attention_params(attn_mode, split_attn)
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.out_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x: torch.Tensor, rope_angles: torch.Tensor | None = None) -> torch.Tensor:
        batch, sequence, _ = x.shape
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        q = self.q_norm(q.reshape(batch, sequence, self.heads, self.head_dim))
        k = self.k_norm(k.reshape(batch, sequence, self.heads, self.head_dim))
        v = v.reshape(batch, sequence, self.heads, self.head_dim)
        if rope_angles is not None:
            q = apply_split_half_rope(q, rope_angles)
            k = apply_split_half_rope(k, rope_angles)
        return self.out_proj(common_attention([q, k, v], attn_params=self.attn_params))


class MLP(nn.Module):
    def __init__(self, hidden: int, ffn_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn_hidden * 2, bias=False)
        self.fc2 = nn.Linear(ffn_hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * value)


class AdalnProj(nn.Module):
    def __init__(self, time_dim: int, hidden: int, expand: int, modalities: int, apply_silu: bool = True):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(time_dim, expand * hidden * modalities, bias=True)

    def forward(self, time_embedding: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.apply_silu:
            time_embedding = F.silu(time_embedding)
        projected = self.linear(time_embedding)
        projected = projected.view(projected.shape[0] * self.modalities, self.expand * self.hidden)
        return projected.chunk(self.expand, dim=-1)


class RefinerBlock(nn.Module):
    def __init__(
        self, hidden: int, heads: int, head_dim: int, ffn: int, eps: float, qk_eps: float, attn_mode: str, split_attn: bool
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, attn_mode, split_attn)
        self.mlp = MLP(hidden, ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TokenRefiner(nn.Module):
    def __init__(
        self,
        layers: int,
        hidden: int,
        heads: int,
        head_dim: int,
        ffn: int,
        eps: float,
        qk_eps: float,
        final_eps: float,
        attn_mode: str,
        split_attn: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps, attn_mode, split_attn) for _ in range(layers)]
        )
        self.final_norm = nn.RMSNorm(hidden, eps=final_eps)

    def forward(self, x: torch.Tensor, gradient_checkpointing: bool = False) -> torch.Tensor:
        for block in self.blocks:
            if gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        heads: int,
        head_dim: int,
        ffn: int,
        time_dim: int,
        eps: float,
        qk_eps: float,
        attn_mode: str,
        split_attn: bool,
        apply_silu: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, attn_mode, split_attn)
        self.mlp = MLP(hidden, ffn)
        self.adaln_proj = AdalnProj(time_dim, hidden, expand=6, modalities=3, apply_silu=apply_silu)

    def forward(
        self, x: torch.Tensor, time_embedding: torch.Tensor, modulation_rows: torch.Tensor, rope_angles: torch.Tensor
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(time_embedding)
        shift_msa = shift_msa[modulation_rows][None].to(x.dtype)
        scale_msa = scale_msa[modulation_rows][None].to(x.dtype)
        gate_msa = gate_msa[modulation_rows][None].to(x.dtype)
        hidden = self.norm1(x) * (1.0 + scale_msa) + shift_msa
        x = x + self.attn(hidden, rope_angles) * gate_msa

        shift_mlp = shift_mlp[modulation_rows][None].to(x.dtype)
        scale_mlp = scale_mlp[modulation_rows][None].to(x.dtype)
        gate_mlp = gate_mlp[modulation_rows][None].to(x.dtype)
        hidden = self.norm2(x) * (1.0 + scale_mlp) + shift_mlp
        return x + self.mlp(hidden) * gate_mlp


class FinalLayer(nn.Module):
    def __init__(self, hidden: int, time_dim: int, video_dim: int, audio_dim: int, eps: float, apply_silu: bool = True):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=eps)
        self.adaln_proj = AdalnProj(time_dim, hidden, expand=2, modalities=1, apply_silu=apply_silu)
        self.video_out = nn.Linear(hidden, video_dim, bias=True, dtype=torch.float32)
        self.audio_out = nn.Linear(hidden, audio_dim, bias=True, dtype=torch.float32)

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        video_slice: tuple[int, int],
        audio_slice: tuple[int, int],
        video_time_row: int,
        audio_time_row: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(time_embedding)
        normalized = self.norm(x)
        video = normalized[:, video_slice[0] : video_slice[1]]
        audio = normalized[:, audio_slice[0] : audio_slice[1]]
        video = video * (1.0 + scale[video_time_row].to(video.dtype)) + shift[video_time_row].to(video.dtype)
        audio = audio * (1.0 + scale[audio_time_row].to(audio.dtype)) + shift[audio_time_row].to(audio.dtype)
        return self.video_out(video.float()), self.audio_out(audio.float())


class MiniMaxH3Model(nn.Module):
    """Trainable H3 Omni Transformer for the FL2VA checkpoint."""

    def __init__(
        self,
        hidden_size: int = 5376,
        num_layers: int = 50,
        token_refiner_num_layers: int = 2,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        ffn_hidden_size: int = 14336,
        latents_dim: int = 24,
        audio_latents_dim: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        timestep_input_dim: int = 256,
        time_embed_hidden_size: int = 5376,
        time_embed_dim: int = 2688,
        rope_inv_freq_len: int = 16,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        adaln_curve_grid: int | None = None,
        sigma_shift_video: float = VIDEO_SIGMA_SHIFT,
        sigma_shift_audio: float = AUDIO_SIGMA_SHIFT,
        attn_mode: str = "torch",
        split_attn: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = tuple(patch_size)
        self.latents_dim = latents_dim
        self.audio_latents_dim = audio_latents_dim
        self.sigma_shift_video = sigma_shift_video
        self.sigma_shift_audio = sigma_shift_audio
        self.use_adaln_curves = adaln_curve_grid is not None

        video_patch_dim = latents_dim * math.prod(self.patch_size)
        self.video_patch_proj = nn.Linear(video_patch_dim, hidden_size, bias=True, dtype=torch.float32)
        self.audio_patch_proj = nn.Linear(audio_latents_dim, hidden_size, bias=True, dtype=torch.float32)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True)
        if self.use_adaln_curves:
            self.register_buffer("adaln_t_table", torch.empty(adaln_curve_grid, time_embed_dim, dtype=torch.float32))
        else:
            self.time_embedder = TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim)
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", torch.empty(rope_inv_freq_len, dtype=torch.float32))
        self.token_refiner = TokenRefiner(
            token_refiner_num_layers,
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            ffn_hidden_size,
            norm_eps,
            qk_norm_eps,
            final_norm_eps,
            attn_mode,
            split_attn,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_hidden_size,
                    time_embed_dim,
                    norm_eps,
                    qk_norm_eps,
                    attn_mode,
                    split_attn,
                    apply_silu=not self.use_adaln_curves,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size,
            time_embed_dim,
            video_patch_dim,
            audio_latents_dim,
            final_norm_eps,
            apply_silu=not self.use_adaln_curves,
        )

        self.gradient_checkpointing = False
        self.blocks_to_swap = 0
        self.offloader = None

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def enable_block_swap(self, num_blocks: int, config: BlockSwapConfig):
        if num_blocks > len(self.blocks) - 2:
            raise ValueError(f"Cannot swap more than {len(self.blocks) - 2} H3 blocks; requested {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = create_offloader("single", self.blocks, len(self.blocks), num_blocks, config)

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            saved_blocks = self.blocks
            self.blocks = nn.ModuleList()
        self.to(device)
        if self.blocks_to_swap:
            self.blocks = saved_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(self.blocks)

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def rope_angles(self, position_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        positions = position_ids.to(device=device, dtype=torch.float32)
        inv_freq = self.rope.inv_freq.to(device=device)
        per_axis = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        temporal, height, width = per_axis.unbind(dim=1)
        half = torch.cat([temporal, height, width], dim=-1)
        return torch.cat([half, half], dim=-1)

    def _forward_single(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        sigma_video: torch.Tensor,
        context: torch.Tensor,
        text_token_tags: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video.shape[0] != 1 or audio.shape[0] != 1 or context.shape[0] != 1:
            raise ValueError("_forward_single requires batch size 1")
        if video.shape[-2] % self.patch_size[1] or video.shape[-1] % self.patch_size[2]:
            raise ValueError("H3 video latent height and width must be divisible by the spatial patch size")

        latent_t, latent_h, latent_w = video.shape[-3:]
        audio_t = audio.shape[-1]
        text_len = context.shape[1]
        layout = PackedLayout(text_len, latent_t, latent_h, latent_w, audio_t)

        sigma_v = sigma_video.flatten()[0].float().clamp(1e-6, 1.0)
        sigma_a = time_shift_sigma(sigma_v, self.sigma_shift_video, self.sigma_shift_audio)
        t_video = float(1.0 - sigma_v)
        t_audio = float(1.0 - sigma_a)
        unique_times = sorted({t_video, t_audio})
        time_row = {value: index for index, value in enumerate(unique_times)}

        if text_token_tags is None:
            text_token_tags = torch.ones(text_len, device=video.device, dtype=torch.long)
        else:
            text_token_tags = text_token_tags.reshape(-1).to(device=video.device, dtype=torch.long)
            if text_token_tags.shape[0] != text_len:
                raise ValueError("text_token_tags length must match the text context length")

        modulation_rows = torch.empty(layout.seq_len, device=video.device, dtype=torch.long)
        modulation_rows[:text_len] = time_row[t_video] * 3 + text_token_tags
        modulation_rows[layout.audio_slice[0] : layout.audio_slice[1]] = time_row[t_audio] * 3 + 2
        modulation_rows[layout.video_slice[0] : layout.video_slice[1]] = time_row[t_video] * 3

        compute_dtype = context.dtype
        video_rows = patchify_video(video.float(), self.patch_size)
        audio_rows = pack_audio(audio.float())
        video_embed = self.video_patch_proj(video_rows).to(compute_dtype)
        audio_embed = self.audio_patch_proj(audio_rows).to(compute_dtype)

        if context.shape[-1] != self.hidden_size:
            context = self.token_refiner(self.condition_proj(context), self.gradient_checkpointing)
        hidden = torch.cat([context, audio_embed, video_embed], dim=1)

        time_values = torch.tensor(unique_times, device=video.device, dtype=torch.float32)
        if self.use_adaln_curves:
            table = self.adaln_t_table.to(device=video.device)
            position = time_values.clamp(0.0, 1.0) * (table.shape[0] - 1)
            lower = position.floor().long().clamp(max=table.shape[0] - 2)
            time_embedding = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
        else:
            time_embedding = self.time_embedder(time_values).to(compute_dtype)
        rope_angles = self.rope_angles(layout.position_ids, video.device)

        for index, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    block, hidden, time_embedding, modulation_rows, rope_angles, use_reentrant=False
                )
            else:
                hidden = block(hidden, time_embedding, modulation_rows, rope_angles)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, index)

        video_rows, audio_rows = self.final_layer(
            hidden,
            time_embedding,
            layout.video_slice,
            layout.audio_slice,
            time_row[t_video],
            time_row[t_audio],
        )
        video_out = unpatchify_video(
            video_rows,
            latent_t,
            latent_h // self.patch_size[1],
            latent_w // self.patch_size[2],
            self.latents_dim,
            self.patch_size,
        )
        return video_out.to(video.dtype), unpack_audio(audio_rows).to(audio.dtype)

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        sigma_video: torch.Tensor,
        context: torch.Tensor | list[torch.Tensor],
        text_token_tags: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw checkpoint predictions (``clean - noise``) for both streams.

        H3's packed sequence has shape-dependent length and the released full
        attention implementation is batch-one.  Larger loader batches are
        evaluated serially while preserving one accumulated autograd graph.
        """

        outputs_video: list[torch.Tensor] = []
        outputs_audio: list[torch.Tensor] = []
        for index in range(video.shape[0]):
            context_i = context[index]
            if context_i.ndim == 2:
                context_i = context_i.unsqueeze(0)
            tags_i = None
            if text_token_tags is not None:
                tags_i = text_token_tags[index]
            video_i, audio_i = self._forward_single(
                video[index : index + 1],
                audio[index : index + 1],
                sigma_video[index : index + 1],
                context_i,
                tags_i,
            )
            outputs_video.append(video_i)
            outputs_audio.append(audio_i)
        return torch.cat(outputs_video, dim=0), torch.cat(outputs_audio, dim=0)
