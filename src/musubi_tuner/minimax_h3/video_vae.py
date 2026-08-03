"""Encoder-only MiniMax H3 video VAE used by latent caching."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.minimax_h3.minimax_h3_utils import load_selected_weights, resolve_safetensor_files


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
)


class CausalConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

    def forward(self, x):
        if sum(self.causal_padding) == 0:
            return super().forward(x)
        pt, ph, pw = self.causal_padding
        x = F.pad(x, (pw, pw, ph, ph, 0, 0), mode="reflect")
        if x.shape[2] == 1:
            # The preceding causal taps multiply zero padding. Avoid materializing them.
            return F.conv3d(x, self.weight[:, :, -1:], self.bias, self.stride, 0, self.dilation, self.groups)
        x = F.pad(x, (0, 0, 0, 0, pt * 2, 0))
        return super().forward(x)


class TemporalIsolatedGroupNorm(nn.GroupNorm):
    def forward(self, x):
        if x.ndim != 5:
            return super().forward(x)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, 1, h, w)
        return super().forward(x).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def group_norm_3d(channels):
    return TemporalIsolatedGroupNorm(32, channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(
            in_channels, out_channels, 3, padding=(1, 0, 0), stride=(time_stride, space_stride, space_stride)
        )

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = group_norm_3d(in_channels)
        self.norm2 = group_norm_3d(self.out_channels)
        self.conv1 = CausalConv3d(in_channels, self.out_channels, 3, padding=1)
        self.conv2 = CausalConv3d(self.out_channels, self.out_channels, 3, padding=1)
        if in_channels != self.out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, self.out_channels, 1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return h + x


class EncoderFCN3D(nn.Module):
    def __init__(
        self,
        ch=128,
        ch_mult=(1, 2, 2, 4, 4, 8),
        space_down=(2, 2, 2, 2, 1, 1),
        time_down=(1, 2, 2, 1, 1, 1),
        num_res_blocks=2,
        in_channels=3,
        z_channels=24,
    ):
        super().__init__()
        levels = len(ch_mult)
        counts = [num_res_blocks] * levels if isinstance(num_res_blocks, int) else list(num_res_blocks)
        block_mid = [ch * value for value in ch_mult]
        block_in = [block_mid[0], *block_mid[:-1]]
        self.num_res_blocks = counts
        self.conv_in = CausalConv3d(in_channels, block_in[0], 3, padding=1)
        self.down = nn.ModuleList()
        for level in range(levels):
            down = nn.Module()
            down.block = nn.ModuleList(
                [
                    ResnetBlock3D(block_in[level] if index == 0 else block_mid[level], block_mid[level])
                    for index in range(counts[level])
                ]
            )
            if space_down[level] * time_down[level] > 1:
                down.downsample = Downsample3D(
                    block_mid[level], block_mid[level], time_stride=time_down[level], space_stride=space_down[level]
                )
            self.down.append(down)
        self.norm_out = group_norm_3d(block_mid[-1])
        self.conv_out = CausalConv3d(block_mid[-1], 2 * z_channels, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for level, down in enumerate(self.down):
            for block in down.block:
                h = block(h)
            if hasattr(down, "downsample"):
                h = down.downsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class MiniMaxH3VideoEncoder(nn.Module):
    def __init__(self, tile_size=256, tile_overlap=64, tiling=True):
        super().__init__()
        self.vae_ratio = 16
        self.clip_length = 17
        self.token_drop = 3
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap
        self.tiling = tiling
        self.encoder = EncoderFCN3D()
        self.quant_conv = nn.Conv3d(48, 48, 1)
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN), persistent=False)
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD), persistent=False)
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    @property
    def device(self):
        return self.quant_conv.weight.device

    @property
    def dtype(self):
        return self.quant_conv.weight.dtype

    def _encode_moments(self, x):
        return self.quant_conv(self.encoder(x))

    def split_tiles(self, input_len):
        if self.tile_size >= input_len:
            return [0], [input_len], []
        count = math.ceil(input_len / self.tile_size)
        while True:
            overlaps = [self.tile_overlap_min] * (count - 1)
            remaining = self.tile_size * count - sum(overlaps) - input_len
            if remaining >= 0:
                break
            count += 1
        for index in range(remaining // self.vae_ratio):
            overlaps[index % (count - 1)] += self.vae_ratio
        starts = [0]
        for index in range(count - 1):
            starts.append(starts[-1] + self.tile_size - overlaps[index])
        return starts, [self.tile_size] * count, overlaps

    @staticmethod
    def blend(a, b, extent, dim):
        extent = min(a.shape[dim], b.shape[dim], extent)
        if extent == 0:
            return b
        pos = torch.arange(extent, device=b.device, dtype=b.dtype) / extent
        shape = [1] * b.ndim
        shape[dim] = extent
        pos = pos.view(shape)
        a_slice = [slice(None)] * a.ndim
        b_slice = [slice(None)] * b.ndim
        a_slice[dim] = slice(-extent, None)
        b_slice[dim] = slice(0, extent)
        blended = a[tuple(a_slice)] * (1 - pos) + b[tuple(b_slice)] * pos
        if extent == b.shape[dim]:
            return blended
        b_slice[dim] = slice(extent, None)
        return torch.cat([blended, b[tuple(b_slice)]], dim=dim)

    def tiled_encode(self, x):
        y_starts, y_lengths, y_overlaps = self.split_tiles(x.shape[-2])
        x_starts, x_lengths, x_overlaps = self.split_tiles(x.shape[-1])
        rows = []
        for y, yl in zip(y_starts, y_lengths):
            rows.append([self._encode_moments(x[..., y : y + yl, xx : xx + xl]) for xx, xl in zip(x_starts, x_lengths)])
        ly = [value // self.vae_ratio for value in y_overlaps]
        lx = [value // self.vae_ratio for value in x_overlaps]
        output_rows = []
        for i, row in enumerate(rows):
            output = []
            for j, tile in enumerate(row):
                if i:
                    tile = self.blend(rows[i - 1][j], tile, ly[i - 1], -2)
                if j:
                    tile = self.blend(row[j - 1], tile, lx[j - 1], -1)
                if i < len(rows) - 1:
                    tile = tile[..., :-ly[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, :-lx[j]]
                output.append(tile)
            output_rows.append(torch.cat(output, dim=-1))
        return torch.cat(output_rows, dim=-2)

    def _adaptive_encode(self, x):
        return self.tiled_encode(x) if self.tiling else self._encode_moments(x)

    def encode(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x = ((x + 1.0) * 0.5 - self.pixel_mean.to(x)) / self.pixel_std.to(x)
        if x.shape[2] == 1:
            moments = self._adaptive_encode(x)[:, :, -1:]
        else:
            pad = (-x.shape[2]) % self.clip_length
            if pad:
                x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)
            moments = torch.cat(
                [self._adaptive_encode(x[:, :, i : i + self.clip_length]) for i in range(0, x.shape[2], self.clip_length)],
                dim=2,
            )[:, :, : -self.token_drop]
        mean = moments.float().chunk(2, dim=1)[0]
        mean_value = self.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        std_value = self.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return (mean - mean_value) / std_value


def load_video_vae(
    path: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    tile_size: int = 256,
    tile_overlap: int = 64,
    tiling: bool = True,
    disable_numpy_memmap: bool = False,
) -> MiniMaxH3VideoEncoder:
    with torch.device("meta"):
        model = MiniMaxH3VideoEncoder(tile_size, tile_overlap, tiling)
    load_selected_weights(
        model,
        resolve_safetensor_files(path, "video_vae"),
        device=device,
        dtype=dtype,
        disable_numpy_memmap=disable_numpy_memmap,
    )
    # Non-persistent constants were created on meta with the module.
    model.latents_mean = torch.tensor(LATENTS_MEAN, device=device)
    model.latents_std = torch.tensor(LATENTS_STD, device=device)
    model.pixel_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1, 1)
    model.pixel_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1, 1)
    model.eval().requires_grad_(False)
    return model

