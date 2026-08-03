import argparse
import logging

import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_MINIMAX_H3,
    ItemInfo,
    save_latent_cache_minimax_h3,
)
from musubi_tuner.minimax_h3.audio_vae import load_audio_segment, load_audio_vae
from musubi_tuner.minimax_h3.minimax_h3_utils import audio_latent_length
from musubi_tuner.minimax_h3.video_vae import load_video_vae
from musubi_tuner.utils.model_utils import str_to_dtype


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@torch.no_grad()
def encode_and_save_batch(video_vae, audio_vae, batch: list[ItemInfo]):
    for item in batch:
        if item.content is None or item.frame_count is None:
            raise ValueError(f"H3 requires video content for {item.item_key}")
        content = torch.from_numpy(item.content).unsqueeze(0).permute(0, 4, 1, 2, 3).contiguous()
        content = content.to(video_vae.device, dtype=video_vae.dtype) / 127.5 - 1.0
        video_latent = video_vae.encode(content)[0].to(video_vae.dtype)

        if item.source_path is None:
            raise ValueError(f"Original media path is unavailable for {item.item_key}")
        waveform = load_audio_segment(item.source_path, item.source_frame_start, item.frame_count)
        waveform = waveform.unsqueeze(0).to(audio_vae.device, dtype=audio_vae.dtype)
        audio_latent = audio_vae.encode(waveform, audio_latent_length(item.frame_count))[0]
        save_latent_cache_minimax_h3(item, video_latent, audio_latent)


def minimax_h3_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--audio_vae", type=str, required=True, help="H3 audio VAE safetensors file or model directory")
    parser.add_argument("--vae_tile_size", type=int, default=256, help="video VAE spatial tile size")
    parser.add_argument("--vae_tile_overlap", type=int, default=64, help="video VAE spatial tile overlap")
    parser.add_argument("--disable_vae_tiling", action="store_true", help="disable video VAE spatial tiling")
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="disable memory-mapped checkpoint loading")
    return parser


def main():
    parser = minimax_h3_setup_parser(cache_latents.setup_parser_common())
    args = parser.parse_args()
    if args.vae is None:
        parser.error("--vae is required for MiniMax H3 latent caching")
    if args.disable_cudnn_backend:
        torch.backends.cudnn.enabled = False
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        config_utils.load_user_config(args.dataset_config), args, architecture=ARCHITECTURE_MINIMAX_H3
    )
    datasets = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets
    for dataset in datasets:
        if getattr(dataset, "source_fps", None) is None:
            logger.warning(
                "MiniMax H3 assumes unresampled source frames are already 24 fps. Set source_fps to the actual "
                "video rate when it differs, or cached video and audio may be out of sync."
            )
    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=24
        )
        return

    dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("MiniMax H3 VAE caching requires float16, bfloat16, or float32")
    logger.info("Loading MiniMax H3 video VAE")
    video_vae = load_video_vae(
        args.vae,
        device=device,
        dtype=dtype,
        tile_size=args.vae_tile_size,
        tile_overlap=args.vae_tile_overlap,
        tiling=not args.disable_vae_tiling,
        disable_numpy_memmap=args.disable_numpy_memmap,
    )
    logger.info("Loading MiniMax H3 audio VAE")
    audio_vae = load_audio_vae(
        args.audio_vae, device=device, dtype=dtype, disable_numpy_memmap=args.disable_numpy_memmap
    )

    cache_latents.encode_datasets(datasets, lambda batch: encode_and_save_batch(video_vae, audio_vae, batch), args)


if __name__ == "__main__":
    main()
