import argparse
import logging

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_MINIMAX_H3,
    ItemInfo,
    save_text_encoder_output_cache_minimax_h3,
)
from musubi_tuner.minimax_h3.text_encoder import encode_prompts, load_text_encoder, load_tokenizer
from musubi_tuner.utils.model_utils import str_to_dtype


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@torch.no_grad()
def encode_and_save_batch(tokenizer, text_encoder, batch: list[ItemInfo], device, max_length):
    embeds, tags = encode_prompts(tokenizer, text_encoder, [item.caption for item in batch], device, max_length)
    for item, embed, token_tags in zip(batch, embeds, tags):
        save_text_encoder_output_cache_minimax_h3(item, embed, token_tags)


def minimax_h3_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="H3 Qwen3-VL safetensors file or model directory")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer directory; defaults to the model bundle/Hugging Face")
    parser.add_argument("--text_encoder_dtype", type=str, default="bfloat16", help="text encoder dtype")
    parser.add_argument("--max_token_length", type=int, default=1024, help="maximum raw prompt token length")
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="disable memory-mapped checkpoint loading")
    return parser


def main():
    parser = minimax_h3_setup_parser(cache_text_encoder_outputs.setup_parser_common())
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        config_utils.load_user_config(args.dataset_config), args, architecture=ARCHITECTURE_MINIMAX_H3
    )
    datasets = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets
    all_files, all_paths = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    tokenizer = load_tokenizer(args.tokenizer, args.text_encoder)
    dtype = str_to_dtype(args.text_encoder_dtype)
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("MiniMax H3 text encoding requires float16, bfloat16, or float32")
    logger.info("Loading MiniMax H3 Qwen3-VL text encoder (layers 0-49)")
    text_encoder = load_text_encoder(
        args.text_encoder, device=device, dtype=dtype, disable_numpy_memmap=args.disable_numpy_memmap
    )
    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_files,
        all_paths,
        lambda batch: encode_and_save_batch(tokenizer, text_encoder, batch, device, args.max_token_length),
    )
    cache_text_encoder_outputs.post_process_cache_files(datasets, all_files, all_paths, args.keep_cache)


if __name__ == "__main__":
    main()
